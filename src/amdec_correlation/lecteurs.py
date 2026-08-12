"""Aiguillage des formats d'acquisition.

Le pipeline ne connaît qu'une chose des fichiers d'essai : un **lecteur** sait
décrire leurs canaux et charger des signaux sur une base de temps commune. Le
format concret — MDF4 aujourd'hui, d'autres ensuite — est isolé derrière cette
interface.

Ajouter un format revient à écrire deux fonctions et à les enregistrer :

    from .lecteurs import Lecteur, enregistrer

    enregistrer(Lecteur(
        nom="Mon format",
        extensions=(".ext",),
        decrire=ma_fonction_decrire,     # (chemin) -> list[DescriptionCanal]
        charger=ma_fonction_charger,     # (chemin, mapping, frequence) -> SignauxEssai
    ))

Rien d'autre ne change : inventaire, mapping, traitements et rapport
fonctionnent à l'identique quel que soit le format d'origine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

__all__ = [
    "Lecteur",
    "MappingLibre",
    "enregistrer",
    "lecteur_pour",
    "extensions_supportees",
    "formats_supportes",
    "lister_fichiers",
    "doublons_de_format",
    "decrire_canaux",
    "charger_signaux",
    "charger_canaux",
    "FormatNonSupporte",
]


class FormatNonSupporte(RuntimeError):
    """Extension sans lecteur enregistré. Le message liste les formats connus."""


@dataclass(frozen=True)
class Lecteur:
    """Un format d'acquisition et les deux fonctions qui permettent de l'exploiter."""

    nom: str
    extensions: tuple[str, ...]
    decrire: Callable[..., list]
    charger: Callable[..., object]

    def __post_init__(self) -> None:
        if not self.extensions:
            raise ValueError(f"Lecteur « {self.nom} » : au moins une extension est requise.")


_LECTEURS: dict[str, Lecteur] = {}


def enregistrer(lecteur: Lecteur) -> None:
    """Déclare un lecteur pour ses extensions (comparaison insensible à la casse)."""
    for extension in lecteur.extensions:
        _LECTEURS[extension.lower()] = lecteur


def lecteur_pour(chemin: str | Path) -> Lecteur:
    """Lecteur associé à l'extension du fichier."""
    extension = Path(chemin).suffix.lower()
    lecteur = _LECTEURS.get(extension)
    if lecteur is None:
        connus = ", ".join(sorted(extensions_supportees())) or "aucun"
        raise FormatNonSupporte(
            f"Aucun lecteur pour l'extension « {extension or '(sans extension)'} » "
            f"({Path(chemin).name}). Formats pris en charge : {connus}."
        )
    return lecteur


def extensions_supportees() -> set[str]:
    return set(_LECTEURS)


def formats_supportes() -> dict[str, tuple[str, ...]]:
    """Nom de format -> extensions, pour l'affichage à l'utilisateur."""
    par_nom: dict[str, tuple[str, ...]] = {}
    for lecteur in set(_LECTEURS.values()):
        par_nom[lecteur.nom] = lecteur.extensions
    return par_nom


def lister_fichiers(dossier: str | Path, max_fichiers: int | None = None) -> list[Path]:
    """Acquisitions exploitables d'un sous-dossier d'essai, tous formats confondus.

    La recherche est récursive et insensible à la casse de l'extension.
    """
    dossier = Path(dossier)
    if not dossier.is_dir():
        return []
    fichiers = sorted(
        {
            chemin
            for chemin in dossier.rglob("*")
            if chemin.is_file() and chemin.suffix.lower() in _LECTEURS
        },
        key=lambda p: p.name.lower(),
    )
    return fichiers[:max_fichiers] if max_fichiers else fichiers


# -- fonctions d'accès, aiguillées vers le lecteur du format -----------------


def doublons_de_format(fichiers: Iterable[Path]) -> dict[str, list[Path]]:
    """Repère les acquisitions présentes sous plusieurs formats.

    Un même essai exporté à la fois en `.mf4` et en `.aif` porte les mêmes
    grandeurs : le traiter deux fois fausserait tout ce qui se cumule — nombre
    de paliers, répétabilité, dispersion des cartes de contrôle — sans qu'aucun
    calcul n'échoue. Le doublon se reconnaît au nom de fichier sans extension.
    """
    par_racine: dict[str, list[Path]] = {}
    for chemin in fichiers:
        par_racine.setdefault(chemin.stem.lower(), []).append(chemin)
    return {racine: sorted(v) for racine, v in par_racine.items() if len(v) > 1}


@dataclass(frozen=True)
class MappingLibre:
    """Présente une liste de canaux quelconques comme un mapping de rôles.

    Les lecteurs chargent des canaux désignés par un *rôle* (couple mesuré,
    régime…). Pour la visualisation on veut au contraire tracer des canaux
    choisis librement, sans leur attribuer de rôle. Cet adaptateur les expose
    sous la forme attendue par les lecteurs — la clé est alors le nom réel du
    canal — ce qui évite de dupliquer la logique de chargement et de
    rééchantillonnage.
    """

    canaux: tuple[str, ...]
    etat: tuple[str, ...] = ()

    def presents(self) -> dict[str, str]:
        return {nom: nom for nom in self.canaux}


def charger_canaux(chemin: str | Path, noms: Iterable[str], frequence_Hz: float):
    """Charge des canaux désignés par leur nom réel, sur une grille commune."""
    return lecteur_pour(chemin).charger(
        chemin, MappingLibre(tuple(dict.fromkeys(noms))), frequence_Hz
    )


def decrire_canaux(chemin: str | Path) -> list:
    """Inventorie les canaux d'un fichier, quel que soit son format."""
    return lecteur_pour(chemin).decrire(chemin)


def charger_signaux(chemin: str | Path, mapping, frequence_Hz: float):
    """Charge les canaux mappés sur une grille de temps commune."""
    return lecteur_pour(chemin).charger(chemin, mapping, frequence_Hz)


def _enregistrer_lecteurs_integres() -> None:
    """Enregistre les lecteurs livrés avec le paquet.

    L'import est fait ici, et non au sommet du module, pour éviter une
    dépendance circulaire : le lecteur MDF4 importe ce module pour s'y déclarer.
    """
    from . import io_mdf  # noqa: F401  (l'import provoque l'enregistrement)


_enregistrer_lecteurs_integres()
