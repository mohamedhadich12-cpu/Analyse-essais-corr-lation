#!/usr/bin/env python3
"""Planche des allures de couple attendues, essai par essai.

    python docs/source/allures_essais.py

Produit `docs/allures_couple_par_essai.png` : pour chacun des sept essais de la
campagne, la forme que doit avoir le couple au cours du temps, et les zones sur
lesquelles le pipeline s'appuie.

⚠️ CE SONT DES ALLURES DE PRINCIPE, tracées à la main pour illustrer chaque type
d'essai. Aucune donnée mesurée n'y figure. La planche sert à vérifier d'un coup
d'œil que le type déclaré pour un dossier correspond à ce que contiennent
réellement les fichiers — pas à valider des valeurs.

Charte reprise des figures d'analyse : une seule série par panneau, marques
fines, grille en filet plein, encres de texte jamais colorées.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FOND, ENCRE, ENCRE_2, ATTENUE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRILLE, AXE = "#e1e0d9", "#c3c2b7"
SERIE_1, AQUA, ORANGE = "#2a78d6", "#1baf7a", "#eb6834"
SORTIE = Path(__file__).resolve().parents[1] / "allures_couple_par_essai.png"

ALEA = np.random.default_rng(7)


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": FOND, "axes.facecolor": FOND, "savefig.facecolor": FOND,
        "font.family": "sans-serif", "font.size": 8,
        "axes.titlesize": 9, "axes.titleweight": "bold",
        "axes.labelsize": 7.5, "axes.labelcolor": ENCRE_2,
        "axes.edgecolor": AXE, "axes.linewidth": .8,
        "xtick.color": ATTENUE, "ytick.color": ATTENUE,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "grid.color": GRILLE, "grid.linewidth": .6, "grid.linestyle": "-",
        "legend.frameon": False, "legend.fontsize": 7,
        "lines.linewidth": 1.4, "figure.dpi": 160, "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })


def _habiller(ax, titre: str, sous_titre: str) -> None:
    ax.grid(True, zorder=0)
    ax.set_axisbelow(True)
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)
    ax.set_title(titre, color=ENCRE, loc="left", pad=13)
    ax.text(0, 1.015, sous_titre, transform=ax.transAxes, color=ATTENUE, fontsize=7,
            va="bottom")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Couple (N·m)")


def _annoter(ax, x, y, texte, dx=0, dy=0) -> None:
    ax.annotate(
        texte, xy=(x, y), xytext=(x + dx, y + dy), color=ENCRE_2, fontsize=6.8,
        ha="center", va="center",
        arrowprops=dict(arrowstyle="-", color=ATTENUE, linewidth=.7,
                        shrinkA=0, shrinkB=3),
        bbox=dict(facecolor=FOND, edgecolor=GRILLE, linewidth=.5,
                  boxstyle="round,pad=0.25"),
    )


def _escalier(niveaux, duree_palier=8.0, duree_rampe=2.0, fe=20.0):
    """Suite de paliers reliés par des rampes."""
    t, y, horloge = [], [], 0.0
    for i, niveau in enumerate(niveaux):
        n = int(duree_palier * fe)
        t.append(horloge + np.arange(n) / fe); y.append(np.full(n, float(niveau)))
        horloge += duree_palier
        if i < len(niveaux) - 1:
            m = int(duree_rampe * fe)
            t.append(horloge + np.arange(m) / fe)
            y.append(np.linspace(niveau, niveaux[i + 1], m))
            horloge += duree_rampe
    return np.concatenate(t), np.concatenate(y)


def _bruit(y, amplitude=3.0):
    return y + ALEA.normal(0, amplitude, y.size)


# ---------------------------------------------------------------------------
# Les sept allures
# ---------------------------------------------------------------------------


def balayage(ax) -> None:
    niveaux = [0, 200, 400, 600, 800, 1000, 1200]
    t, y = _escalier(niveaux + niveaux[-2::-1])
    ax.plot(t, _bruit(y), color=SERIE_1, zorder=3)
    sommet = t[np.argmax(y)]
    ax.axvspan(t[0], sommet, color=AQUA, alpha=.07, zorder=1)
    ax.text(sommet / 2, 1290, "MONTÉE", color=ENCRE_2, fontsize=7, ha="center", weight="bold")
    ax.text((sommet + t[-1]) / 2, 1290, "DESCENTE", color=ENCRE_2, fontsize=7,
            ha="center", weight="bold")
    _annoter(ax, 45, 600, "chaque palier tenu\njusqu'à établissement", dx=-2, dy=-330)
    ax.set_ylim(-90, 1450)
    _habiller(ax, "1-Balayage Couple", "type « balayage » — l'aller-retour est indispensable")


def repetabilite(ax) -> None:
    niveaux = [0, 250, 500, 750, 1000]
    decalages = [0, 14, -9]   # trois relevés successifs du même protocole
    horloge = 0.0
    for i, decalage in enumerate(decalages):
        t, y = _escalier(niveaux, duree_palier=6.0, duree_rampe=1.5)
        ax.plot(t + horloge, _bruit(y + decalage), color=SERIE_1,
                alpha=1.0 if i == 0 else .55, zorder=3)
        if i:
            ax.axvline(horloge, color=AXE, linewidth=.8, zorder=2)
        horloge += t[-1] + 4.0
    _annoter(ax, 40, 1000, "les MÊMES niveaux,\nrépétés relevé après relevé", dx=42, dy=-330)
    ax.set_ylim(-90, 1250)
    _habiller(ax, "2-CPC 20 °C", "type « répétabilité » — la répétition des niveaux fait tout")


PUISSANCE_W = 125_700.0  # ≈ 1000 N·m à 1200 tr/min, ordre de grandeur d'une traction


def _couple_iso_puissance(regime_tr_min):
    """Couple à puissance constante : C = P / ω, avec ω = 2π·N/60."""
    return PUISSANCE_W / (np.asarray(regime_tr_min) * 2 * np.pi / 60.0)


def iso_puissance_statique(ax) -> None:
    # À puissance constante, le couple suit l'inverse du régime : les paliers
    # sont stabilisés mais tous à des couples DIFFÉRENTS.
    regimes = [1200, 1500, 1900, 2400, 3100, 4000]
    couples = _couple_iso_puissance(regimes)
    t, y = _escalier(list(couples), duree_palier=9.0, duree_rampe=2.0)
    ax.plot(t, _bruit(y, 3.0), color=SERIE_1, zorder=3)
    for regime, couple in zip(regimes[::2], couples[::2]):
        ax.text(2, couple + 28, f"{regime} tr/min", color=ATTENUE, fontsize=6.2, va="bottom")
    _annoter(ax, 26, float(couples[2]), "iso-puissance :\ncouple × régime constant",
             dx=26, dy=300)
    ax.set_ylim(0, 1200)
    _habiller(ax, "4-iso Pwr traction MaV statique",
              "type « répétabilité » — paliers stabilisés, mais à couples tous différents")


def depart_arrete(ax) -> None:
    fe, duree = 50.0, 60.0
    t = np.arange(0, duree, 1 / fe)
    y = np.zeros_like(t)
    lance = (t >= 10) & (t < 14)
    y[lance] = 1150 * (1 - np.exp(-(t[lance] - 10) / 0.6))
    roule = t >= 14
    y[roule] = 1150 * np.exp(-(t[roule] - 14) / 26) + 60
    y[t > 52] = 0.0
    ax.plot(t, _bruit(y, 6), color=SERIE_1, zorder=3)
    ax.axvspan(0, 10, color=AQUA, alpha=.10, zorder=1)
    ax.text(5, 1180, "zéro", color=ENCRE_2, fontsize=6.8, ha="center")
    _annoter(ax, 11.5, 900, "front très raide :\nle meilleur support\npour le retard",
             dx=17, dy=180)
    ax.set_ylim(-90, 1400)
    _habiller(ax, "3-4_1000M_DA — départ arrêté",
              "type « dynamique » — transitoire franc depuis l'arrêt")


def decollage_pente(ax) -> None:
    fe, duree = 50.0, 70.0
    t = np.arange(0, duree, 1 / fe)
    y = np.zeros_like(t)
    montee = (t >= 12) & (t < 18)
    y[montee] = 1250 * (1 - np.exp(-(t[montee] - 12) / 1.6))
    tenue = (t >= 18) & (t < 50)
    y[tenue] = 1180 + 70 * np.sin(2 * np.pi * (t[tenue] - 18) / 9)
    relache = (t >= 50) & (t < 58)
    y[relache] = 1180 * np.exp(-(t[relache] - 50) / 2.2)
    ax.plot(t, _bruit(y, 8), color=SERIE_1, zorder=3)
    ax.axvspan(0, 12, color=AQUA, alpha=.10, zorder=1)
    ax.axvspan(58, duree, color=AQUA, alpha=.10, zorder=1)
    ax.text(6, 1420, "zéro début", color=ENCRE_2, fontsize=6.8, ha="center")
    ax.text(64, 1420, "zéro fin", color=ENCRE_2, fontsize=6.8, ha="center")
    _annoter(ax, 34, 1180, "couple élevé maintenu\n→ forte montée en température", dx=0, dy=-620)
    ax.set_ylim(-90, 1620)
    _habiller(ax, "5-Décollage en pente",
              "type « dynamique » — les deux relevés de zéro donnent la dérive")


def iso_puissance_dynamique(ax) -> None:
    fe, duree = 50.0, 80.0
    t = np.arange(0, duree, 1 / fe)
    # Régime borné à la même plage que l'essai statique : sans borne, un régime
    # proche de zéro ferait diverger C = P/ω et créerait un pic sans réalité.
    regime = 2600 + 1100 * np.sin(2 * np.pi * t / 22) + 300 * np.sin(2 * np.pi * t / 6.5)
    y = _couple_iso_puissance(np.clip(regime, 1200, 4000))
    ax.plot(t, _bruit(y, 4), color=SERIE_1, zorder=3)
    _annoter(ax, 47, float(y.min()), "mêmes points de fonctionnement\nque l'essai statique,\n"
             "parcourus en continu", dx=-2, dy=430)
    ax.set_ylim(150, 1250)
    _habiller(ax, "6-iso Pwr traction MaV dynamique",
              "type « dynamique » — le pendant transitoire de l'essai statique")


def wltc(ax) -> None:
    fe, duree = 25.0, 300.0
    t = np.arange(0, duree, 1 / fe)
    y = (420 + 330 * np.sin(2 * np.pi * t / 62)
         + 180 * np.sin(2 * np.pi * t / 21 + 1.1)
         + 90 * np.sin(2 * np.pi * t / 7.5 + .4))
    repos = (t < 20) | (t > duree - 20)
    y[repos] = 0.0
    ax.plot(t, _bruit(y, 6), color=SERIE_1, zorder=3)
    ax.axvspan(0, 20, color=AQUA, alpha=.10, zorder=1)
    ax.axvspan(duree - 20, duree, color=AQUA, alpha=.10, zorder=1)
    ax.axvspan(20, duree - 20, color=ORANGE, alpha=.06, zorder=1)
    ax.text(duree / 2, 1010, "plage exploitée pour l'intercorrélation",
            color=ENCRE_2, fontsize=6.8, ha="center")
    ax.set_ylim(-380, 1150)
    _habiller(ax, "7-WLTC", "type « dynamique » — variations permanentes, phases contrastées")


def cle_de_lecture(ax) -> None:
    ax.axis("off")
    ax.text(0, 1.0, "Comment s'en servir", transform=ax.transAxes, color=ENCRE,
            fontsize=9, weight="bold", va="top")
    lignes = [
        ("balayage", "Des paliers, et un aller-retour.\n"
                     "Sans descente, pas d'hystérésis."),
        ("répétabilité", "Des paliers aux MÊMES niveaux, répétés.\n"
                         "À iso-puissance, prévoir plusieurs relevés."),
        ("dynamique", "Des variations continues.\n"
                      "Un signal trop plat masque le retard."),
    ]
    # La position du bloc suivant est calée sur l'ENCOMBREMENT RÉEL du bloc
    # précédent, mesuré après tracé. Estimer la hauteur d'une ligne conduit
    # immanquablement à recouvrir la légende du bas.
    figure = ax.get_figure()
    figure.canvas.draw()
    y = 0.90
    for type_essai, texte in lignes:
        ax.text(0, y, type_essai, transform=ax.transAxes, color=SERIE_1,
                fontsize=7.6, weight="bold", va="top")
        bloc = ax.text(0, y - 0.055, texte, transform=ax.transAxes, color=ENCRE_2,
                       fontsize=7, va="top", linespacing=1.6)
        figure.canvas.draw()
        boite = bloc.get_window_extent().transformed(ax.transAxes.inverted())
        y = boite.y0 - 0.045

    for hauteur, couleur, alpha, libelle in (
        (0.09, AQUA, .25, "relevé de zéro (dérive)"),
        (0.03, ORANGE, .18, "plage dynamique (retard)"),
    ):
        ax.add_patch(plt.Rectangle((0, hauteur), 0.06, 0.035, transform=ax.transAxes,
                                   facecolor=couleur, alpha=alpha, edgecolor="none"))
        ax.text(0.08, hauteur + 0.017, libelle, transform=ax.transAxes,
                color=ENCRE_2, fontsize=7, va="center")


def construire() -> Path:
    _style()
    figure, axes = plt.subplots(4, 2, figsize=(11.6, 13.2))
    figure.suptitle(
        "Allure du couple attendue par type d'essai",
        x=0.09, y=0.982, ha="left", fontsize=14, color=ENCRE, weight="bold",
    )
    figure.text(
        0.09, 0.962,
        "Allures de principe, tracées à titre d'illustration — aucune donnée mesurée. "
        "Sert à vérifier que le type déclaré correspond au contenu réel des fichiers.",
        ha="left", fontsize=8, color=ATTENUE,
    )
    for trace, ax in zip(
        (balayage, repetabilite, iso_puissance_statique, depart_arrete,
         decollage_pente, iso_puissance_dynamique, wltc, cle_de_lecture),
        axes.ravel(),
    ):
        trace(ax)
    figure.subplots_adjust(hspace=0.52, wspace=0.22, top=0.935)
    figure.savefig(SORTIE)
    plt.close(figure)
    return SORTIE


if __name__ == "__main__":
    chemin = construire()
    print(f"Planche écrite : {chemin}")
