"""Lecture **strictement en lecture seule** des acquisitions MDF4.

Garantie de non-modification des sources : chaque fichier est ouvert via un
handle Python en mode binaire `'rb'` passé à `asammdf.MDF`. L'écriture est donc
impossible au niveau du système de fichiers, et aucune méthode mutante
d'asammdf (`save`, `cut`, `convert`, ...) n'est appelée par ce module.

Ce module gère aussi le problème de base de temps : dans un MDF4, chaque groupe
de canaux possède son propre canal maître, avec sa propre fréquence et son
propre instant d'origine. Les canaux ne partagent donc pas nécessairement la
même base de temps, et une comparaison échantillon à échantillon exige un
rééchantillonnage sur une grille commune — fait ici explicitement, avec un
diagnostic restitué à l'utilisateur.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
from asammdf import MDF

from .config import MappingCanaux

# Extensions traitées comme des conteneurs MDF. `.aif` en fait partie : les
# exports ETAS INCA sont des conteneurs MDF sous une extension propre à l'outil.
# L'appartenance réelle au format est vérifiée sur le CONTENU du fichier
# (cf. `_verifier_conteneur_mdf`), jamais sur son nom.
EXTENSIONS_MDF = (".mf4", ".mdf", ".aif")

# Signatures d'un bloc d'identification MDF, versions 2 à 4 confondues.
# « UnFinMF » marque un fichier non finalisé, que la bibliothèque sait réparer.
SIGNATURES_MDF = (b"MDF     ", b"UnFinMF ")


class CanalIntrouvable(KeyError):
    """Canal absent du fichier : le message liste les canaux réellement présents."""


class ErreurChargement(RuntimeError):
    """Le fichier ne peut pas être exploité (canal vide, pas de plage commune...)."""


def _verifier_conteneur_mdf(chemin: Path) -> None:
    """Vérifie que le fichier est bien un conteneur MDF, sur son contenu.

    Une extension ne prouve rien. Sans ce contrôle, un fichier au format
    inattendu produirait soit une erreur incompréhensible de la bibliothèque,
    soit — bien pire — des signaux plausibles mais faux. On préfère refuser
    explicitement, en donnant de quoi identifier le format réel.
    """
    with open(chemin, "rb") as fh:
        entete = fh.read(8)
    if entete.startswith(SIGNATURES_MDF):
        return
    raise ErreurChargement(
        f"{chemin.name} n'est pas un conteneur MDF : ses premiers octets sont "
        f"{entete.hex(' ')} ({entete!r}), au lieu de « MDF     ». "
        "L'extension seule ne garantit pas le format. Si ce fichier provient bien "
        "d'un outil de mesure, communiquez ces premiers octets pour qu'un lecteur "
        "adapté soit écrit — aucune donnée ne sera interprétée à l'aveugle."
    )


@contextmanager
def ouvrir_mdf(chemin: str | Path) -> Iterator[MDF]:
    """Ouvre un conteneur MDF en lecture seule.

    Le handle est ouvert en `'rb'` : toute tentative d'écriture par la
    bibliothèque échouerait au niveau OS. C'est la garantie technique que les
    acquisitions sources ne sont jamais altérées.
    """
    chemin = Path(chemin)
    _verifier_conteneur_mdf(chemin)
    with open(chemin, "rb") as fh:
        mdf = MDF(fh)
        try:
            yield mdf
        finally:
            mdf.close()


def lister_fichiers(dossier: str | Path, max_fichiers: int | None = None) -> list[Path]:
    """Acquisitions d'un sous-dossier, tous formats enregistrés confondus.

    Conservé ici par commodité ; l'implémentation vit dans `lecteurs`, qui seul
    connaît la liste des formats pris en charge.
    """
    from .lecteurs import lister_fichiers as _lister

    return _lister(dossier, max_fichiers)


# ---------------------------------------------------------------------------
# Inventaire brut des canaux
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DescriptionCanal:
    nom: str
    unite: str
    groupe: int
    index: int
    n_echantillons: int
    frequence_Hz: float | None
    t_debut: float | None
    t_fin: float | None
    commentaire: str = ""

    @property
    def duree_s(self) -> float | None:
        if self.t_debut is None or self.t_fin is None:
            return None
        return self.t_fin - self.t_debut


def decrire_canaux(chemin: str | Path) -> list[DescriptionCanal]:
    """Inventorie tous les canaux d'un fichier : nom, unité, cadence, plage de temps.

    C'est la brique de l'étape 1 : elle sert à confirmer les libellés réels
    avant de renseigner le mapping, plutôt que de supposer une convention.
    """
    descriptions: list[DescriptionCanal] = []
    with ouvrir_mdf(chemin) as mdf:
        for i_groupe, groupe in enumerate(mdf.groups):
            # Le canal maître (temps) porte l'échantillonnage du groupe.
            try:
                maitre = mdf.get_master(i_groupe)
            except Exception:  # groupe sans maître exploitable
                maitre = np.array([])

            n = int(maitre.size)
            t0 = float(maitre[0]) if n else None
            t1 = float(maitre[-1]) if n else None
            freq = None
            if n > 1 and t1 is not None and t0 is not None and t1 > t0:
                freq = (n - 1) / (t1 - t0)

            for i_canal, canal in enumerate(groupe.channels):
                descriptions.append(
                    DescriptionCanal(
                        nom=canal.name,
                        unite=(canal.unit or "").strip(),
                        groupe=i_groupe,
                        index=i_canal,
                        n_echantillons=n,
                        frequence_Hz=freq,
                        t_debut=t0,
                        t_fin=t1,
                        commentaire=(canal.comment or "").strip(),
                    )
                )
    return descriptions


# ---------------------------------------------------------------------------
# Chargement sur base de temps commune
# ---------------------------------------------------------------------------


@dataclass
class DiagnosticTemps:
    """Constat sur les bases de temps des canaux effectivement chargés."""

    par_canal: dict[str, dict[str, float]] = field(default_factory=dict)
    bases_identiques: bool = True
    ecart_origines_s: float = 0.0
    fenetre_commune_s: float = 0.0

    def resume(self) -> str:
        if self.bases_identiques:
            return (
                f"Base de temps commune à tous les canaux "
                f"(fenêtre exploitable {self.fenetre_commune_s:.1f} s)."
            )
        return (
            f"Bases de temps distinctes selon les canaux "
            f"(écart d'origine max {self.ecart_origines_s * 1e3:.1f} ms) ; "
            f"rééchantillonnage sur grille commune, fenêtre exploitable "
            f"{self.fenetre_commune_s:.1f} s."
        )


@dataclass
class SignauxEssai:
    """Signaux d'un fichier, rééchantillonnés sur une grille de temps commune."""

    fichier: Path
    t: np.ndarray
    scalaires: dict[str, np.ndarray]
    etats: dict[str, np.ndarray]
    unites: dict[str, str]
    diagnostic: DiagnosticTemps
    frequence_Hz: float

    def a(self, nom_logique: str) -> np.ndarray | None:
        return self.scalaires.get(nom_logique)

    @property
    def duree_s(self) -> float:
        return float(self.t[-1] - self.t[0]) if self.t.size > 1 else 0.0


def _resoudre(mdf: MDF, nom: str, chemin: Path) -> tuple[int, int]:
    """Localise un canal par son nom, en signalant les homonymes."""
    occurrences = mdf.whereis(nom)
    if not occurrences:
        disponibles = sorted({c for c in mdf.channels_db})
        apercu = ", ".join(disponibles[:40])
        suite = f" ... (+{len(disponibles) - 40} autres)" if len(disponibles) > 40 else ""
        raise CanalIntrouvable(
            f"Canal « {nom} » absent de {chemin.name}. Canaux présents : {apercu}{suite}"
        )
    if len(occurrences) > 1:
        warnings.warn(
            f"{chemin.name} : le canal « {nom} » existe dans {len(occurrences)} groupes "
            f"{list(occurrences)} ; la première occurrence est utilisée.",
            stacklevel=2,
        )
    return occurrences[0]


def _interpoler(t_cible: np.ndarray, t_source: np.ndarray, y: np.ndarray, discret: bool) -> np.ndarray:
    """Rééchantillonne y(t_source) sur t_cible.

    Interpolation linéaire pour les grandeurs physiques continues ; maintien de
    la dernière valeur (`previous`) pour les canaux d'état, où une
    interpolation linéaire créerait des valeurs qui n'existent pas.
    """
    if t_source.size == 0:
        return np.full(t_cible.shape, np.nan)
    if t_source.size == 1:
        return np.full(t_cible.shape, y[0])
    if discret:
        idx = np.searchsorted(t_source, t_cible, side="right") - 1
        idx = np.clip(idx, 0, t_source.size - 1)
        return y[idx]
    return np.interp(t_cible, t_source, y)


def charger_signaux(
    chemin: str | Path,
    mapping: MappingCanaux,
    frequence_Hz: float,
) -> SignauxEssai:
    """Charge les canaux mappés et les ramène sur une grille de temps commune.

    HYPOTHÈSES :
      * la grille commune couvre l'intersection des plages de temps des canaux
        chargés (aucune extrapolation hors de la plage d'un canal) ;
      * elle est uniforme à `frequence_Hz`, ce qui est requis pour
        l'intercorrélation et les statistiques glissantes ;
      * les canaux physiques sont interpolés linéairement, les canaux d'état
        par maintien de la dernière valeur.
    """
    chemin = Path(chemin)
    a_charger: dict[str, str] = {
        logique: reel for logique, reel in mapping.presents().items() if reel
    }
    if not a_charger:
        raise ErreurChargement(f"{chemin.name} : aucun canal mappé à charger.")

    bruts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    unites: dict[str, str] = {}
    etats_bruts: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    with ouvrir_mdf(chemin) as mdf:
        for logique, reel in a_charger.items():
            g, i = _resoudre(mdf, reel, chemin)
            sig = mdf.get(reel, group=g, index=i)
            bruts[logique] = (
                np.asarray(sig.timestamps, dtype=float),
                np.asarray(sig.samples, dtype=float),
            )
            unites[logique] = (sig.unit or "").strip()
        for reel in mapping.etat:
            try:
                g, i = _resoudre(mdf, reel, chemin)
            except CanalIntrouvable:
                warnings.warn(f"{chemin.name} : canal d'état « {reel} » absent, ignoré.", stacklevel=2)
                continue
            sig = mdf.get(reel, group=g, index=i)
            echantillons = np.asarray(sig.samples)
            if echantillons.dtype.kind not in "iufb":
                echantillons = echantillons.astype(str)
            etats_bruts[reel] = (np.asarray(sig.timestamps, dtype=float), echantillons)
            unites[reel] = (sig.unit or "").strip()

    # Diagnostic de base de temps AVANT rééchantillonnage.
    diagnostic = DiagnosticTemps()
    origines, fins, references = [], [], []
    for logique, (ts, ys) in bruts.items():
        if ts.size == 0:
            raise ErreurChargement(f"{chemin.name} : canal « {a_charger[logique]} » vide.")
        freq = (ts.size - 1) / (ts[-1] - ts[0]) if ts.size > 1 and ts[-1] > ts[0] else float("nan")
        diagnostic.par_canal[logique] = {
            "t_debut": float(ts[0]),
            "t_fin": float(ts[-1]),
            "n": int(ts.size),
            "frequence_Hz": float(freq),
        }
        origines.append(ts[0])
        fins.append(ts[-1])
        references.append(ts)

    diagnostic.ecart_origines_s = float(max(origines) - min(origines))
    diagnostic.bases_identiques = all(
        r.shape == references[0].shape and np.allclose(r, references[0])
        for r in references[1:]
    )

    t_debut, t_fin = float(max(origines)), float(min(fins))
    if t_fin <= t_debut:
        raise ErreurChargement(
            f"{chemin.name} : les canaux mappés n'ont aucune plage de temps commune "
            f"(début max {t_debut:.3f} s > fin min {t_fin:.3f} s)."
        )
    diagnostic.fenetre_commune_s = t_fin - t_debut

    pas = 1.0 / frequence_Hz
    t = np.arange(t_debut, t_fin + 0.5 * pas, pas)
    t = t[t <= t_fin]

    scalaires = {
        logique: _interpoler(t, ts, ys, discret=False) for logique, (ts, ys) in bruts.items()
    }
    etats = {
        nom: _interpoler(t, ts, ys, discret=True) if ys.dtype.kind in "iufb" else _interpoler_str(t, ts, ys)
        for nom, (ts, ys) in etats_bruts.items()
    }

    return SignauxEssai(
        fichier=chemin,
        t=t,
        scalaires=scalaires,
        etats=etats,
        unites=unites,
        diagnostic=diagnostic,
        frequence_Hz=frequence_Hz,
    )


def _interpoler_str(t_cible: np.ndarray, t_source: np.ndarray, y: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(t_source, t_cible, side="right") - 1
    idx = np.clip(idx, 0, t_source.size - 1)
    return y[idx]


def resumer_etats(signaux: SignauxEssai) -> dict[str, dict[str, float]]:
    """Distribution des valeurs de chaque canal d'état.

    La valeur « nominale » d'une LED télémétrie ou d'un compteur CRC n'étant pas
    connue du pipeline, on restitue la distribution brute (part de chaque valeur)
    sans l'interpréter : c'est à l'exploitant de dire quelle valeur est nominale.
    """
    resume: dict[str, dict[str, float]] = {}
    for nom, valeurs in signaux.etats.items():
        uniques, comptes = np.unique(valeurs, return_counts=True)
        total = float(comptes.sum())
        resume[nom] = {
            str(v): round(100.0 * c / total, 3) for v, c in zip(uniques, comptes)
        }
    return resume


# ---------------------------------------------------------------------------
# Déclaration du format auprès de l'aiguillage
# ---------------------------------------------------------------------------

def _enregistrer() -> None:
    """Déclare MDF4 comme format lisible.

    L'import est local pour éviter la circularité : `lecteurs` importe ce module
    afin de provoquer cet enregistrement.
    """
    from .lecteurs import Lecteur, enregistrer

    enregistrer(
        Lecteur(
            nom="MDF (ASAM) — .mf4, .mdf, .aif (export ETAS INCA)",
            extensions=EXTENSIONS_MDF,
            decrire=decrire_canaux,
            charger=charger_signaux,
        )
    )


_enregistrer()
