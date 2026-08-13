"""Découverte automatique des zones exploitables dans une acquisition.

Principe : **on ne classe pas les fichiers, on extrait leurs zones.** Plutôt que
de demander à l'utilisateur de déclarer qu'un dossier contient des balayages et
un autre des cycles, chaque fichier est examiné pour ce qu'il contient
réellement :

  * des **paliers stabilisés** — le couple de référence y reste dans une
    tolérance assez longtemps ; ils alimentent régression, non-linéarité,
    hystérésis et répétabilité ;
  * une **plage dynamique** — le signal y varie assez pour que l'intercorrélation
    identifie un retard ;
  * des **plages de repos** — couple et régime quasi nuls ; une au début et une à
    la fin donnent la dérive du zéro.

Un même fichier peut contenir les trois, et c'est fréquent : un cycle qui débute
et s'achève à l'arrêt fournit ses zéros, sa plage dynamique, et parfois quelques
paliers. Extraire les zones plutôt que d'étiqueter le fichier exploite tout ce
qui est présent, au lieu du seul aspect que l'étiquette aurait retenu.

Ce que la détection ne peut PAS deviner reste à déclarer par l'utilisateur :
la symétrie de chargement du banc (indispensable au résidu gauche − droite) et
l'existence d'un démontage/remontage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import metriques as M
from .config import Config


@dataclass
class ZonesFichier:
    """Ce qu'une acquisition contient, et ce qu'elle peut donc alimenter."""

    chemin: Path
    duree_s: float
    paliers: list[M.Palier] = field(default_factory=list)
    plage_dynamique: tuple[int, int] | None = None
    duree_dynamique_s: float = 0.0
    # Les autres plages actives, écartées au profit de la précédente. Les
    # conserver rend l'arbitrage vérifiable : une plage retenue au mauvais
    # endroit ne se repère qu'en voyant aussi celles qui ont été écartées.
    plages_ecartees: list[tuple[int, int]] = field(default_factory=list)
    plages_repos: list[tuple[int, int]] = field(default_factory=list)

    # Les mêmes plages, en SECONDES. Les indices ci-dessus ne valent que pour la
    # grille de temps sur laquelle la détection a tourné : les porter sur un
    # tracé décimé, recadré, ou rééchantillonné à une autre fréquence — ce que
    # fait l'onglet Visualisation — les fait sortir du tableau. Un instant, lui,
    # reste valable sur n'importe quelle grille. Tout affichage passe donc par
    # ces champs ; les indices restent réservés au découpage des signaux.
    plage_dynamique_s: tuple[float, float] | None = None
    plages_ecartees_s: list[tuple[float, float]] = field(default_factory=list)
    plages_repos_s: list[tuple[float, float]] = field(default_factory=list)
    zero_au_debut: bool = False
    zero_a_la_fin: bool = False
    erreur: str | None = None

    # -- ce que le fichier peut alimenter -----------------------------------
    @property
    def niveaux_distincts(self) -> int:
        return len({round(p.reference, 1) for p in self.paliers})

    @property
    def a_montee_et_descente(self) -> bool:
        sens = {p.sens for p in self.paliers}
        return "montee" in sens and "descente" in sens

    @property
    def alimente_regression(self) -> bool:
        """Trois niveaux distincts au moins : en deçà, une droite n'a pas de sens."""
        return self.niveaux_distincts >= 3

    @property
    def alimente_hysteresis(self) -> bool:
        return self.a_montee_et_descente

    @property
    def alimente_retard(self) -> bool:
        return self.plage_dynamique is not None

    @property
    def alimente_derive_zero(self) -> bool:
        return self.zero_au_debut and self.zero_a_la_fin

    def familles(self) -> list[str]:
        """Familles de zones réellement présentes, dans l'ordre de lecture.

        Sert à ne proposer à l'affichage que ce qui existe : offrir « paliers —
        descente » sur un cycle qui n'en comporte pas ferait douter du réglage
        plutôt que du contenu du fichier.

        Les clés correspondent à celles de `graphiques.TYPES_ZONES`.
        """
        presentes: list[str] = []
        sens = {p.sens for p in self.paliers}
        for cle, condition in (
            ("montee", "montee" in sens),
            ("descente", "descente" in sens),
            ("indetermine", bool(sens - {"montee", "descente"})),
            ("dynamique", self.plage_dynamique is not None),
            ("ecartee", bool(self.plages_ecartees)),
            ("repos", bool(self.plages_repos)),
        ):
            if condition:
                presentes.append(cle)
        return presentes

    def contributions(self) -> list[str]:
        contributions = []
        if self.alimente_regression:
            contributions.append("régression")
        elif self.paliers:
            contributions.append("répétabilité")
        if self.alimente_hysteresis:
            contributions.append("hystérésis")
        if self.alimente_retard:
            contributions.append("retard")
        if self.alimente_derive_zero:
            contributions.append("dérive de zéro")
        return contributions

    def profil(self) -> str:
        """Libellé lisible de ce qui a été trouvé, pour vérification à l'écran."""
        if self.erreur:
            return f"non exploité — {self.erreur}"
        morceaux = []
        if self.paliers:
            montees = sum(1 for p in self.paliers if p.sens == "montee")
            descentes = len(self.paliers) - montees
            detail = f"{len(self.paliers)} paliers"
            if self.a_montee_et_descente:
                detail += f" ({montees} ↑ / {descentes} ↓)"
            morceaux.append(detail)
        if self.plage_dynamique is not None:
            detail = f"dynamique {self.duree_dynamique_s:.0f} s"
            if self.plages_ecartees:
                detail += f" (retenue sur {len(self.plages_ecartees) + 1} candidates)"
            morceaux.append(detail)
        if self.alimente_derive_zero:
            morceaux.append("zéros début et fin")
        elif self.plages_repos:
            morceaux.append(f"{len(self.plages_repos)} plage(s) de repos")
        return " · ".join(morceaux) or "aucune zone exploitable"


def detecter(donnees, cfg: Config) -> ZonesFichier:
    """Examine une acquisition déjà chargée et y repère les zones exploitables."""
    t, reference, mesure = donnees.t, donnees.reference, donnees.mesure
    zones = ZonesFichier(
        chemin=donnees.chemin,
        duree_s=float(t[-1] - t[0]) if t.size > 1 else 0.0,
    )

    zones.paliers = M.detecter_paliers(
        t, reference, mesure, cfg.pleine_echelle_Nm, cfg.paliers,
        temperature=donnees.temperature, source=donnees.chemin.name,
    )

    def _en_secondes(plage: tuple[int, int]) -> tuple[float, float]:
        return (float(t[plage[0]]), float(t[plage[1] - 1]))

    candidates = M.plages_dynamiques(
        t, reference, cfg.pleine_echelle_Nm, cfg.intercorrelation
    )
    if candidates:
        plage = candidates[0]
        zones.plage_dynamique = plage
        zones.plage_dynamique_s = _en_secondes(plage)
        zones.duree_dynamique_s = float(t[plage[1] - 1] - t[plage[0]])
        zones.plages_ecartees = candidates[1:]
        zones.plages_ecartees_s = [_en_secondes(p) for p in candidates[1:]]

    zones.plages_repos = M.plages_de_repos(
        t, reference, donnees.regime, cfg.pleine_echelle_Nm, cfg.zero
    )
    zones.plages_repos_s = [_en_secondes(p) for p in zones.plages_repos]
    if zones.plages_repos and zones.duree_s > 0:
        bord = cfg.zero.fraction_bord * zones.duree_s
        premiere, derniere = zones.plages_repos[0], zones.plages_repos[-1]
        zones.zero_au_debut = bool(t[premiere[0]] <= t[0] + bord)
        zones.zero_a_la_fin = bool(t[derniere[1] - 1] >= t[-1] - bord)
        # Une seule plage de repos couvrant tout l'essai ne constitue pas deux
        # relevés : il en faut deux distinctes pour parler de dérive.
        if len(zones.plages_repos) < 2:
            zones.zero_au_debut = zones.zero_a_la_fin = False

    return zones


def niveaux_repetes(
    zones: list[ZonesFichier], cfg: Config
) -> dict[float, list[M.Palier]]:
    """Regroupe les paliers de TOUS les fichiers par niveau de couple.

    C'est ce regroupement inter-fichiers qui rend la répétabilité calculable sans
    déclaration : deux acquisitions passant par le même point de fonctionnement
    constituent une répétition, qu'elles proviennent ou non du même protocole.
    """
    tolerance = cfg.paliers.tolerance_appariement_pc_pe * cfg.pleine_echelle_Nm / 100.0
    tous = sorted(
        (p for z in zones for p in z.paliers), key=lambda p: p.reference
    )
    if not tous:
        return {}
    groupes: list[list[M.Palier]] = [[tous[0]]]
    for palier in tous[1:]:
        if abs(palier.reference - groupes[-1][0].reference) <= tolerance:
            groupes[-1].append(palier)
        else:
            groupes.append([palier])
    return {
        float(np.mean([p.reference for p in g])): g
        for g in groupes
        if len({p.source for p in g}) >= 2  # répété par au moins deux fichiers
    }


def synthese(zones: list[ZonesFichier]) -> dict[str, int]:
    """Compte, par grandeur, le nombre de fichiers qui peuvent l'alimenter."""
    return {
        "fichiers": len(zones),
        "exploités": sum(1 for z in zones if z.erreur is None),
        "régression": sum(1 for z in zones if z.alimente_regression),
        "hystérésis": sum(1 for z in zones if z.alimente_hysteresis),
        "retard": sum(1 for z in zones if z.alimente_retard),
        "dérive de zéro": sum(1 for z in zones if z.alimente_derive_zero),
        "paliers au total": sum(len(z.paliers) for z in zones),
    }
