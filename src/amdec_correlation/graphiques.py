"""Génération des figures PNG destinées à l'insertion directe dans le rapport.

Charte graphique (palette validée, mode clair unique — les figures sont
destinées à un document imprimé) :

  * une couleur = une entité, jamais un rang. Sur les tracés temporels la
    référence banc est toujours bleue et le couple mesuré par les transmissions
    toujours orange ; sur le balayage, la montée est bleue et la descente
    orange. Le lecteur apprend l'association une fois pour toute la campagne.
  * les droites de régression sont tracées en gris : ce sont des *modèles*, pas
    des entités mesurées, elles ne consomment donc pas de couleur de série.
  * grille et axes en filet plein, une nuance au-dessus du fond ; jamais de
    pointillés, qui se liraient comme un seuil.
  * marques fines, marqueurs cerclés de la couleur du fond pour se détacher
    lorsqu'ils se recouvrent ; légende systématique dès deux séries.
  * les textes (valeurs, annotations, axes) portent les encres de texte et
    jamais la couleur d'une série.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # rendu fichier, sans serveur graphique
import matplotlib.pyplot as plt
import numpy as np

from .metriques import (
    Hysteresis,
    Palier,
    Recalage,
    Regression,
    Repetabilite,
    SensibiliteThermique,
    appliquer_retard,
)

# -- jetons de la palette ---------------------------------------------------
FOND = "#fcfcfb"
ENCRE = "#0b0b0b"
ENCRE_2 = "#52514e"
ATTENUE = "#898781"
GRILLE = "#e1e0d9"
AXE = "#c3c2b7"

SERIE_1 = "#2a78d6"  # bleu   : référence banc / montée
SERIE_2 = "#eb6834"  # orange : couple mesuré transmissions / descente
SERIE_3 = "#1baf7a"  # aqua   : troisième série, toujours étiquetée en clair
BLEU_CLAIR = "#86b6ef"  # nuage de points brut (même famille de teinte)

DPI = 150


def _appliquer_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": FOND,
            "axes.facecolor": FOND,
            "savefig.facecolor": FOND,
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.labelcolor": ENCRE_2,
            "axes.edgecolor": AXE,
            "axes.linewidth": 0.8,
            "text.color": ENCRE,
            "xtick.color": ATTENUE,
            "ytick.color": ATTENUE,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "grid.color": GRILLE,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",  # filet plein : jamais de pointillés
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.6,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
        }
    )


def _habiller(ax, titre: str = "", x: str = "", y: str = "") -> None:
    """Chrome récessif : grille en filet, deux montants d'axe seulement."""
    ax.grid(True, which="major", axis="both", zorder=0)
    ax.set_axisbelow(True)
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)
    for cote in ("left", "bottom"):
        ax.spines[cote].set_color(AXE)
    if titre:
        ax.set_title(titre, color=ENCRE, loc="left", pad=8)
    if x:
        ax.set_xlabel(x)
    if y:
        ax.set_ylabel(y)


def _annoter(ax, texte: str, position: str = "upper left") -> None:
    """Encadré de valeurs, en encre de texte (jamais en couleur de série)."""
    coords = {
        "upper left": (0.02, 0.98, "left", "top"),
        "lower right": (0.98, 0.03, "right", "bottom"),
        "upper right": (0.98, 0.98, "right", "top"),
    }[position]
    ax.text(
        coords[0],
        coords[1],
        texte,
        transform=ax.transAxes,
        ha=coords[2],
        va=coords[3],
        color=ENCRE_2,
        fontsize=8,
        linespacing=1.5,
        bbox=dict(facecolor=FOND, edgecolor=GRILLE, linewidth=0.6, boxstyle="round,pad=0.45"),
    )


def _legende_hors_trace(ax, ncol: int = 2) -> None:
    """Pose la légende sur la ligne du titre, hors de la zone tracée.

    Sur un tracé temporel, la courbe occupe toute la largeur : une légende
    posée à l'intérieur des axes finit toujours par recouvrir des données,
    quel que soit le coin choisi.
    """
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=ncol, borderaxespad=0.0)
    # Le titre est aligné à gauche sur la même bande : on le remonte au-dessus
    # de la légende pour qu'ils ne se chevauchent jamais. `get_title()` sans
    # argument lirait le titre CENTRÉ, toujours vide ici.
    titre = ax.get_title(loc="left")
    if titre:
        ax.set_title(titre, color=ENCRE, loc="left", pad=26)


def _marge_y(ax, haut: float = 0.16, bas: float = 0.16) -> None:
    """Dégage du blanc au-dessus et au-dessous des courbes (encadrés de valeurs)."""
    y0, y1 = ax.get_ylim()
    etendue = y1 - y0
    ax.set_ylim(y0 - bas * etendue, y1 + haut * etendue)


def _points(ax, x, y, couleur, label, marqueur="o"):
    """Marqueurs cerclés du fond : 2 px de séparation quand ils se recouvrent."""
    return ax.plot(
        x, y, marqueur, color=couleur, markersize=6, markeredgecolor=FOND,
        markeredgewidth=1.2, linestyle="none", label=label, zorder=3,
    )


# ---------------------------------------------------------------------------
# 1. Régression du balayage couple
# ---------------------------------------------------------------------------


def figure_regression_balayage(
    paliers: list[Palier],
    reg: Regression,
    hyst: Hysteresis,
    pleine_echelle_Nm: float,
    chemin: Path,
    titre: str = "Balayage couple — corrélation transmissions / banc GMP",
) -> Path:
    """Régression des points stabilisés + résidus, en deux panneaux liés."""
    _appliquer_style()
    fig, (haut, bas) = plt.subplots(
        2, 1, figsize=(7.0, 5.6), sharex=True, height_ratios=[3, 1.35],
        gridspec_kw={"hspace": 0.12},
    )

    montees = [p for p in paliers if p.sens == "montee"]
    descentes = [p for p in paliers if p.sens == "descente"]

    if not reg.non_calculable:
        xs = np.linspace(
            min(p.reference for p in paliers), max(p.reference for p in paliers), 200
        )
        haut.plot(xs, reg.a * xs + reg.b, color=ATTENUE, linewidth=1.3,
                  label="régression linéaire", zorder=2)

    if montees:
        _points(haut, [p.reference for p in montees], [p.mesure for p in montees],
                SERIE_1, "montée", "o")
    if descentes:
        _points(haut, [p.reference for p in descentes], [p.mesure for p in descentes],
                SERIE_2, "descente", "^")

    _habiller(haut, titre, "", "Couple mesuré transmissions (N·m)")
    haut.legend(loc="lower right")

    if not reg.non_calculable:
        lignes = [
            f"a = {reg.a:.4f} ± {reg.sigma_a:.4f}",
            f"b = {reg.b:+.2f} ± {reg.sigma_b:.2f} N·m",
            f"soit {100 * reg.b / pleine_echelle_Nm:+.2f} % PE",
            f"R² = {reg.r2:.5f}   (n = {reg.n} paliers)",
        ]
        if not hyst.non_calculable:
            lignes.append(f"hystérésis max = {hyst.max_pc_pe:.3f} % PE")
        _annoter(haut, "\n".join(lignes))

    # Panneau bas : résidus par rapport à la droite, en % PE.
    if not reg.non_calculable:
        residus_pc = 100.0 * reg.residus / pleine_echelle_Nm
        sens = [p.sens for p in paliers][: residus_pc.size]
        bas.axhline(0, color=AXE, linewidth=0.8, zorder=1)
        for etiquette, couleur, marqueur in (
            ("montee", SERIE_1, "o"), ("descente", SERIE_2, "^")
        ):
            idx = [i for i, s in enumerate(sens) if s == etiquette]
            if idx:
                _points(bas, reg.x[idx], residus_pc[idx], couleur, None, marqueur)
        marge = max(float(np.max(np.abs(residus_pc))) * 1.4, 0.05)
        bas.set_ylim(-marge, marge)

    _habiller(bas, "", "Couple de référence banc GMP (N·m)", "Résidu (% PE)")
    fig.savefig(chemin)
    plt.close(fig)
    return chemin


# ---------------------------------------------------------------------------
# 2. Résidu en fonction de la température
# ---------------------------------------------------------------------------


def _cadrer_sur_le_nuage(ax, sens_th, pleine_echelle_Nm: float) -> None:
    """Cadre l'axe des résidus sur les centiles 1–99 du nuage.

    Quelques transitoires isolés suffisent à écraser toute la structure du
    nuage si l'axe les englobe. Le cadrage est **purement visuel** : aucun
    point n'est écarté du calcul de la régression, et l'encadré de valeurs
    signale les points sortis du cadre.
    """
    if not sens_th.nuage_res.size:
        return
    valeurs = 100.0 * sens_th.nuage_res / pleine_echelle_Nm
    bas, haut = np.percentile(valeurs, [1, 99])
    if sens_th.residus.size:  # les moyennes par classe restent toujours visibles
        classes = 100.0 * sens_th.residus / pleine_echelle_Nm
        bas, haut = min(bas, classes.min()), max(haut, classes.max())
    marge = max((haut - bas) * 0.25, 0.05)
    ax.set_ylim(bas - marge, haut + marge)
    hors_cadre = int(np.sum((valeurs < bas - marge) | (valeurs > haut + marge)))
    if hors_cadre:
        ax.text(
            0.02, 0.02,
            f"{hors_cadre} échantillon(s) hors cadre — conservés dans le calcul",
            transform=ax.transAxes, ha="left", va="bottom", color=ATTENUE, fontsize=7.5,
        )


def figure_residu_temperature(
    sens_th: SensibiliteThermique,
    pleine_echelle_Nm: float,
    chemin: Path,
    titre: str = "Résidu (mesuré − référence) en fonction de la température",
) -> Path:
    _appliquer_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    if sens_th.nuage_T.size:
        pas = max(1, sens_th.nuage_T.size // 4000)  # allège le rendu, garde la forme
        ax.plot(
            sens_th.nuage_T[::pas], 100.0 * sens_th.nuage_res[::pas] / pleine_echelle_Nm,
            "o", color=BLEU_CLAIR, markersize=2.5, linestyle="none", alpha=0.45,
            label="échantillons", zorder=2,
        )
    if sens_th.temperatures.size:
        _points(
            ax, sens_th.temperatures, 100.0 * sens_th.residus / pleine_echelle_Nm,
            SERIE_1, "moyenne par classe de température",
        )
    if not sens_th.non_calculable and sens_th.temperatures.size:
        xs = np.linspace(sens_th.temperatures.min(), sens_th.temperatures.max(), 100)
        # La droite est reconstruite à partir de la pente et du barycentre des classes.
        y0 = float(np.mean(100.0 * sens_th.residus / pleine_echelle_Nm))
        x0 = float(np.mean(sens_th.temperatures))
        ax.plot(xs, y0 + sens_th.pc_pe_par_C * (xs - x0), color=ATTENUE,
                linewidth=1.3, label="régression linéaire", zorder=4)
        lignes = [
            f"sensibilité = {sens_th.pc_pe_par_C:+.4f} % PE/°C",
            f"soit {sens_th.pc_pe_pour_10C:+.3f} % PE pour 10 °C",
            f"R² = {sens_th.r2:.3f}   ({sens_th.n_classes} cellules)",
            f"excursion = {sens_th.amplitude_C:.1f} °C",
        ]
        if sens_th.correction_couple:
            lignes.append(
                f"régression multiple, effet du couple retiré\n"
                f"(coefficient {sens_th.coefficient_couple:+.4f} N·m/N·m)"
            )
        _annoter(ax, "\n".join(lignes))
    elif sens_th.non_calculable:
        _annoter(ax, sens_th.non_calculable.replace(" : ", " :\n"))

    ax.axhline(0, color=AXE, linewidth=0.8, zorder=1)
    ordonnee = (
        "Résidu corrigé de l'effet couple (% PE)"
        if sens_th.correction_couple
        else "Résidu (% PE)"
    )
    _habiller(ax, titre, "Température (°C)", ordonnee)
    _cadrer_sur_le_nuage(ax, sens_th, pleine_echelle_Nm)
    ax.legend(loc="lower right")
    fig.savefig(chemin)
    plt.close(fig)
    return chemin


# ---------------------------------------------------------------------------
# 3. Superposition avant / après recalage temporel
# ---------------------------------------------------------------------------


def figure_recalage(
    t: np.ndarray,
    reference: np.ndarray,
    mesure: np.ndarray,
    recal: Recalage,
    chemin: Path,
    titre: str = "Recalage temporel",
    duree_zoom_s: float = 20.0,
) -> Path:
    """Trois panneaux : avant recalage, après recalage, et pic d'intercorrélation.

    Le zoom est centré sur la fenêtre effectivement utilisée pour
    l'intercorrélation, sur `duree_zoom_s` secondes : à l'échelle du cycle
    complet, un décalage de quelques dizaines de millisecondes serait invisible.
    """
    _appliquer_style()
    fig, (avant, apres, correl) = plt.subplots(
        3, 1, figsize=(7.0, 8.4), height_ratios=[2, 2, 1.5],
        gridspec_kw={"hspace": 0.72},
    )

    if recal.non_calculable:
        centre = 0.5 * (t[0] + t[-1])
    else:
        centre = 0.5 * (recal.fenetre[0] + recal.fenetre[1])
    debut = max(t[0], centre - duree_zoom_s / 2)
    fin = min(t[-1], debut + duree_zoom_s)
    fenetre = (t >= debut) & (t <= fin)

    def tracer(ax, y_mesure, sous_titre):
        ax.plot(t[fenetre], reference[fenetre], color=SERIE_1,
                label="référence banc GMP", zorder=3)
        ax.plot(t[fenetre], y_mesure[fenetre], color=SERIE_2,
                label="couple mesuré transmissions", zorder=4)
        _habiller(ax, sous_titre, "Temps (s)", "Couple (N·m)")
        _marge_y(ax, haut=0.10, bas=0.28)  # place pour l'encadré de valeurs
        _legende_hors_trace(ax, ncol=2)

    tracer(avant, mesure, f"{titre} — avant recalage")

    if not recal.non_calculable:
        recalee = appliquer_retard(t, mesure, recal.retard_ms / 1000.0)
        tracer(apres, recalee, f"{titre} — après recalage de {recal.retard_ms:+.1f} ms")
        _annoter(
            apres,
            f"RMS résidu : {recal.rms_residu_avant_Nm:.1f} → "
            f"{recal.rms_residu_apres_Nm:.1f} N·m  ({100 * recal.gain_rms:+.0f} %)",
            position="lower right",
        )

        correl.plot(recal.retards_ms, recal.correlation, color=SERIE_1,
                    label="intercorrélation normalisée", zorder=3)
        correl.axvline(recal.retard_ms, color=SERIE_2, linewidth=1.3,
                       label=f"pic : {recal.retard_ms:+.1f} ms", zorder=4)
        _habiller(correl, "Fonction d'intercorrélation", "Décalage appliqué à la mesure (ms)", "r (−)")
        _marge_y(correl, haut=0.12, bas=0.08)
        _legende_hors_trace(correl, ncol=2)
    else:
        tracer(apres, mesure, f"{titre} — recalage non appliqué")
        _annoter(apres, recal.non_calculable.replace(" : ", " :\n"), position="lower right")
        _habiller(correl, "Fonction d'intercorrélation", "Décalage (ms)", "r (−)")
        _annoter(correl, "non calculable")

    fig.savefig(chemin)
    plt.close(fig)
    return chemin


# ---------------------------------------------------------------------------
# 4. Répétabilité
# ---------------------------------------------------------------------------


def figure_repetabilite(
    rep: Repetabilite, pleine_echelle_Nm: float, chemin: Path,
    titre: str = "Répétabilité — dispersion du résidu à couple de référence constant",
) -> Path:
    _appliquer_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    if rep.non_calculable or not rep.groupes:
        _habiller(ax, titre, "Couple de référence banc GMP (N·m)", "Résidu (% PE)")
        _annoter(ax, (rep.non_calculable or "aucun groupe exploitable").replace(" : ", " :\n"))
        fig.savefig(chemin)
        plt.close(fig)
        return chemin

    x = [g.reference_moyenne for g in rep.groupes]
    s = [100.0 * g.ecart_type_residu_Nm / pleine_echelle_Nm for g in rep.groupes]
    ax.axhline(0, color=AXE, linewidth=0.8, zorder=1)
    # ±2σ par niveau : barre d'erreur fine, marqueur cerclé du fond.
    ax.errorbar(
        x, [0] * len(x), yerr=[2 * v for v in s], fmt="o", color=SERIE_1,
        markersize=6, markeredgecolor=FOND, markeredgewidth=1.2,
        elinewidth=1.4, capsize=4, capthick=1.2, label="±2σ par niveau de couple", zorder=3,
    )
    ax.axhline(2 * rep.ecart_type_pc_pe, color=ATTENUE, linewidth=1.1,
               label=f"±2σ poolé = ±{2 * rep.ecart_type_pc_pe:.3f} % PE", zorder=2)
    ax.axhline(-2 * rep.ecart_type_pc_pe, color=ATTENUE, linewidth=1.1, zorder=2)
    _annoter(
        ax,
        f"σ poolé = {rep.ecart_type_pc_pe:.4f} % PE ({rep.ecart_type_Nm:.2f} N·m)\n"
        f"{rep.n_points} paliers, {len(rep.groupes)} niveaux, {rep.degres_liberte} ddl",
    )
    _habiller(ax, titre, "Couple de référence banc GMP (N·m)", "Résidu (% PE)")
    _marge_y(ax, haut=0.30, bas=0.22)
    _legende_hors_trace(ax, ncol=2)
    fig.savefig(chemin)
    plt.close(fig)
    return chemin


# ---------------------------------------------------------------------------
# 5. Redondance gauche / droite
# ---------------------------------------------------------------------------


def figure_redondance(
    t: np.ndarray, gauche: np.ndarray, droite: np.ndarray,
    pleine_echelle_Nm: float, chemin: Path,
    titre: str = "Redondance des voies — résidu gauche − droite",
) -> Path:
    _appliquer_style()
    fig, (haut, bas) = plt.subplots(
        2, 1, figsize=(7.0, 5.2), sharex=True, height_ratios=[2, 1.4],
        gridspec_kw={"hspace": 0.14},
    )
    haut.plot(t, gauche, color=SERIE_1, label="transmission gauche", zorder=3)
    haut.plot(t, droite, color=SERIE_2, label="transmission droite", zorder=4)
    _habiller(haut, titre, "", "Couple (N·m)")
    _marge_y(haut, haut=0.10, bas=0.10)
    _legende_hors_trace(haut, ncol=2)

    ecart = 100.0 * (np.asarray(gauche) - np.asarray(droite)) / pleine_echelle_Nm
    bas.axhline(0, color=AXE, linewidth=0.8, zorder=1)
    bas.plot(t, ecart, color=SERIE_3, linewidth=1.2, label="gauche − droite", zorder=3)
    _habiller(bas, "", "Temps (s)", "Écart (% PE)")
    _marge_y(bas, haut=0.10, bas=0.32)  # place pour l'encadré de valeurs
    # Série 3 (aqua) : sous 3:1 de contraste sur fond clair, la règle de relief
    # impose une identification explicite — d'où la légende et l'encadré chiffré.
    _legende_hors_trace(bas, ncol=1)
    fini = ecart[np.isfinite(ecart)]
    if fini.size:
        _annoter(
            bas,
            f"moyenne {fini.mean():+.3f} % PE — σ {fini.std(ddof=1):.3f} % PE",
            position="lower right",
        )
    fig.savefig(chemin)
    plt.close(fig)
    return chemin


# ---------------------------------------------------------------------------
# 6. Zones détectées dans une acquisition
# ---------------------------------------------------------------------------

# Une teinte par famille de zone, en aplat très clair : la zone est un fond,
# jamais une donnée. Les couleurs de série restent réservées aux courbes.
TEINTE_MONTEE = "#2a78d6"     # bleu   — paliers parcourus en montée
TEINTE_DESCENTE = "#eb6834"   # orange — paliers parcourus en descente
TEINTE_PALIER = "#898781"     # gris   — palier de sens indéterminé
TEINTE_DYNAMIQUE = "#eda100"  # ambre  — plage exploitée pour l'intercorrélation
TEINTE_REPOS = "#1baf7a"      # vert   — relevés de zéro

# Les familles de zones affichables, dans l'ordre de lecture. Chaque clé est
# sélectionnable indépendamment : un balayage porte des dizaines de paliers, et
# les superposer tous à d'autres canaux rend le tracé illisible.
TYPES_ZONES: dict[str, str] = {
    "montee": "paliers — montée",
    "descente": "paliers — descente",
    "indetermine": "paliers — sens indéterminé",
    "dynamique": "plage dynamique (retard)",
    "ecartee": "plages actives écartées",
    "repos": "plages de repos (dérive de zéro)",
}
# Par défaut hors de l'onglet des zones : les familles peu nombreuses. Les
# paliers se demandent explicitement, ce sont eux qui encombrent.
TYPES_ZONES_DEFAUT = ("dynamique", "repos")


def _superposer_zones(ax, t, zones, types: Sequence[str] | None = None) -> dict:
    """Pose les zones détectées en aplats de fond sur un axe temporel.

    `types` restreint l'affichage aux familles demandées (clés de `TYPES_ZONES`) ;
    `None` les affiche toutes. Les zones hors de la fenêtre de temps affichée
    sont ignorées, et celles qui la chevauchent sont rognées.

    Tout est repéré en **secondes**, jamais en indices : le tracé peut être
    décimé, recadré, ou porter une grille de temps différente de celle sur
    laquelle la détection a tourné — un indice n'y survivrait pas, un instant
    si.

    Renvoie les familles effectivement tracées, sous la forme
    `{clé: (teinte ou None, libellé)}` — la teinte vaut `None` pour les hachures,
    qui n'ont pas d'aplat. L'appelant s'en sert pour composer sa légende : une
    couleur sans libellé ne porterait aucune information.
    """
    demandes = set(TYPES_ZONES) if types is None else set(types)
    presents: dict[str, tuple[str | None, str]] = {}
    if zones is None or t.size < 2:
        return presents
    t0, t1 = float(t[0]), float(t[-1])
    etendue = (t1 - t0) or 1.0

    def _aplat(debut: float, fin: float, teinte: str | None, cle: str,
               alpha: float) -> None:
        if cle not in demandes or fin < t0 or debut > t1:
            # Hors de la fenêtre affichée : ne rien tracer, et surtout ne pas
            # inscrire la famille dans la légende — une entrée sans aplat
            # visible ferait chercher au lecteur une zone qui n'est pas là.
            return
        debut, fin = max(debut, t0), min(fin, t1)
        # Une zone de durée nulle à l'écran ne se verrait pas : on garantit une
        # largeur minimale d'un millième de l'axe pour qu'elle reste repérable.
        fin = max(fin, debut + etendue / 1000.0)
        if teinte is None:
            # Plage écartée : hachures, sans aplat, pour se distinguer de la
            # plage retenue sans lui disputer la lisibilité.
            ax.axvspan(debut, fin, facecolor="none", edgecolor=TEINTE_DYNAMIQUE,
                       hatch="///", linewidth=0, alpha=0.55, zorder=1)
        else:
            ax.axvspan(debut, fin, color=teinte, alpha=alpha, linewidth=0, zorder=1)
        presents.setdefault(cle, (teinte, TYPES_ZONES[cle]))

    for palier in getattr(zones, "paliers", []):
        cle = palier.sens if palier.sens in ("montee", "descente") else "indetermine"
        teinte = {"montee": TEINTE_MONTEE, "descente": TEINTE_DESCENTE}.get(
            cle, TEINTE_PALIER
        )
        _aplat(palier.t_debut, palier.t_fin, teinte, cle, alpha=0.22)

    plages = getattr(zones, "plages_dynamiques_s", None)
    if not plages:
        plage = getattr(zones, "plage_dynamique_s", None)
        plages = [plage] if plage is not None else []
    for debut, fin in plages:
        _aplat(debut, fin, TEINTE_DYNAMIQUE, "dynamique", alpha=0.13)

    # Les plages actives écartées : montrer l'arbitrage, et pas seulement son
    # résultat, est le seul moyen de repérer un mauvais choix.
    for debut, fin in getattr(zones, "plages_ecartees_s", []):
        _aplat(debut, fin, None, "ecartee", alpha=0.55)

    for debut, fin in getattr(zones, "plages_repos_s", []):
        _aplat(debut, fin, TEINTE_REPOS, "repos", alpha=0.20)
    return presents


def _poignees_zones(presents: dict) -> tuple[list, list]:
    """Vignettes de légende correspondant aux zones tracées."""
    poignees, etiquettes = [], []
    for teinte, libelle in presents.values():
        if teinte is None:  # plage écartée : hachures, sans aplat
            poignees.append(plt.Rectangle(
                (0, 0), 1, 1, facecolor="none", edgecolor=TEINTE_DYNAMIQUE,
                hatch="///", linewidth=0, alpha=0.55,
            ))
        else:
            poignees.append(plt.Rectangle((0, 0), 1, 1, facecolor=teinte,
                                          alpha=0.30, edgecolor="none"))
        etiquettes.append(libelle)
    return poignees, etiquettes


def figure_zones(
    t: np.ndarray,
    reference: np.ndarray,
    mesure: np.ndarray,
    zones,
    pleine_echelle_Nm: float,
    chemin: Path | None = None,
    titre: str = "",
    decimation: int = 1,
    types: Sequence[str] | None = None,
):
    """Situe sur le signal les zones détectées, chacune identifiée par son type.

    C'est la contrepartie visuelle du tableau de l'onglet « Zones détectées » :
    le tableau dit *combien*, la figure dit *où*. Sans elle, la détection
    resterait à croire sur parole — un palier mal placé, une plage dynamique qui
    déborde sur un transitoire d'arrêt ne se voient que sur le tracé.

    Les zones sont des **aplats de fond**, jamais des courbes : ce sont des
    intervalles de temps, pas des grandeurs mesurées. Chaque type a sa teinte,
    rappelée par une légende explicite — la couleur seule ne porte jamais
    l'information.
    """
    _appliquer_style()
    figure, ax = plt.subplots(figsize=(9.6, 3.9))

    pas = max(1, int(decimation))
    ax.plot(t[::pas], reference[::pas], color=SERIE_1, linewidth=1.1,
            label="couple de référence banc", zorder=3)
    ax.plot(t[::pas], mesure[::pas], color=SERIE_2, linewidth=1.1,
            label="couple mesuré transmissions", zorder=3)

    presents = _superposer_zones(ax, t, zones, types)

    _habiller(ax, titre or zones.chemin.name, "Temps (s)", "Couple (N·m)")
    _marge_y(ax, haut=0.30, bas=0.12)

    poignees, etiquettes = ax.get_legend_handles_labels()
    vignettes, libelles = _poignees_zones(presents)
    poignees += vignettes
    etiquettes += libelles
    ax.legend(poignees, etiquettes, loc="lower right", bbox_to_anchor=(1.0, 1.0),
              ncol=min(3, len(etiquettes)), borderaxespad=0.0)
    ax.set_title(ax.get_title(loc="left"), color=ENCRE, loc="left", pad=32)

    profil = zones.profil()
    alimente = ", ".join(zones.contributions()) or "aucune grandeur"
    note = f"{profil}  ·  alimente : {alimente}"
    if zones.paliers:
        # Sans cette précision, la bande plus étroite que le plateau se lit comme
        # une détection incomplète, alors que c'est le contraire : seule la part
        # établie est moyennée.
        note += ("  ·  les bandes de palier montrent la part effectivement moyennée, "
                 "pas toute la plage stable")
    figure.text(0.012, 0.005, note, ha="left", fontsize=7.5, color=ATTENUE)
    figure.tight_layout(rect=(0, 0.045, 1, 1))
    if chemin is not None:
        figure.savefig(chemin)
        plt.close(figure)
        return chemin
    return figure


# ---------------------------------------------------------------------------
# 7. Visualisation libre de canaux
# ---------------------------------------------------------------------------

# Les huit emplacements de la palette catégorielle, dans leur ordre validé.
# Sur des courbes, cet ordre garantit que deux séries voisines restent
# distinguables, y compris en vision des couleurs déficiente.
SERIES = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)
MAX_COURBES = len(SERIES)


def figure_visualisation(
    t: np.ndarray,
    courbes: dict[str, np.ndarray],
    unites: dict[str, str],
    chemin: Path | None = None,
    titre: str = "",
    decimation: int = 1,
    zones=None,
    types_zones: Sequence[str] | None = None,
):
    """Trace des canaux quelconques en fonction du temps.

    Les courbes sont **groupées par unité**, un panneau par unité, tous alignés
    sur le même axe des temps. C'est le seul tracé correct : superposer un
    couple en N·m et un régime en tr/min sur un axe unique écraserait l'un des
    deux, et leur donner deux échelles verticales inventerait une corrélation
    que les données ne portent pas.

    Une couleur suit un canal d'un panneau à l'autre : le lecteur apprend
    l'association une fois.

    `zones` superpose les zones détectées en aplats de fond, restreintes aux
    familles listées dans `types_zones`. Le choix est laissé à l'appelant parce
    qu'un balayage porte des dizaines de paliers : tout afficher rendrait le
    tracé illisible, et masquer d'office le rendrait incomplet.
    """
    _appliquer_style()
    noms = list(courbes)
    couleurs = {nom: SERIES[i % len(SERIES)] for i, nom in enumerate(noms)}

    groupes: dict[str, list[str]] = {}
    for nom in noms:
        groupes.setdefault(unites.get(nom) or "sans unité", []).append(nom)

    # Bandeau réservé au titre général et à la légende du premier panneau : sans
    # cette réserve, les deux se superposent au titre du panneau.
    BANDEAU = 0.95  # pouces
    hauteur = 1.6 + BANDEAU + 2.15 * len(groupes)
    figure, axes = plt.subplots(
        len(groupes), 1, figsize=(9.6, hauteur), sharex=True, squeeze=False,
        gridspec_kw={"hspace": 0.34},
    )
    axes = axes.ravel()

    presents: dict = {}
    for ax, (unite, membres) in zip(axes, groupes.items()):
        # Les aplats sont posés sur CHAQUE panneau : les panneaux partagent l'axe
        # des temps, une zone vue sur un seul se lirait comme propre à ce canal.
        presents = _superposer_zones(ax, t, zones, types_zones) or presents
        for nom in membres:
            ax.plot(t, courbes[nom], color=couleurs[nom], label=nom, zorder=3)
        # Un seul canal : le titre le nomme, aucune légende n'est nécessaire.
        # Plusieurs : la légende les nomme et l'axe porte l'unité — un titre de
        # panneau ne ferait que répéter l'un ou l'autre.
        _habiller(ax, membres[0] if len(membres) == 1 else "", "", unite)
        _marge_y(ax, haut=0.10, bas=0.10)
        poignees, etiquettes = ax.get_legend_handles_labels()
        if ax is axes[0]:
            # La légende des zones ne figure qu'une fois, sur le premier panneau :
            # répétée, elle mangerait la hauteur utile de chacun.
            vignettes, libelles = _poignees_zones(presents)
            poignees += vignettes
            etiquettes += libelles
        if len(etiquettes) > 1:
            ax.legend(poignees, etiquettes, loc="lower right",
                      bbox_to_anchor=(1.0, 1.0), ncol=min(3, len(etiquettes)),
                      borderaxespad=0.0)
            titre_panneau = ax.get_title(loc="left")
            if titre_panneau:
                ax.set_title(titre_panneau, color=ENCRE, loc="left", pad=26)

    axes[-1].set_xlabel("Temps (s)")
    figure.subplots_adjust(top=1 - BANDEAU / hauteur)
    if titre:
        figure.suptitle(titre, x=0.012, y=1 - 0.22 / hauteur, ha="left", fontsize=11,
                        color=ENCRE, weight="bold")
    if decimation > 1:
        figure.text(
            0.012, 0.002,
            f"Affichage allégé : 1 point sur {decimation}. Les calculs, eux, "
            "portent sur la totalité des échantillons.",
            ha="left", fontsize=7.5, color=ATTENUE,
        )
    if chemin is not None:
        figure.savefig(chemin)
    return figure
