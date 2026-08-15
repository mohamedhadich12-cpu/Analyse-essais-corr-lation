"""Tracé interactif des canaux — zoom, déplacement, curseur de lecture.

Contrepartie manipulable de `graphiques.figure_visualisation`, qui rend une
image fixe. Les deux coexistent volontairement :

* l'**image fixe** part dans le rapport et dans le guide. Elle est reproductible
  au pixel près, ne dépend d'aucun navigateur, et c'est elle qui fait foi ;
* le **tracé interactif** sert à l'examen à l'écran — zoomer sur un front,
  suivre une valeur à l'instant voulu, masquer une courbe d'un clic. Rien de ce
  qu'il montre n'est calculé différemment : ce sont les mêmes tableaux, la même
  palette, et les mêmes zones, rognées par la même fonction.

Une limite à connaître, et elle est affichée à l'utilisateur : le zoom du
navigateur agrandit **les points déjà envoyés**. Il ne fait pas réapparaître
ceux que la décimation a laissés de côté. Pour retrouver la finesse réelle du
signal, il faut resserrer la plage de temps — c'est alors le serveur qui relit
le fichier et décime moins. Laisser croire qu'un zoom restitue des points
absents reviendrait à inventer de la donnée.

Plotly est une dépendance **facultative** : sans elle, l'onglet retombe sur
l'image fixe et le dit. C'est une commodité d'écran, pas un maillon du calcul.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import graphiques as G

try:  # pragma: no cover - dépend de l'environnement d'installation
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _ABSENCE = ""
except Exception as exc:  # noqa: BLE001 - toute cause d'import manqué se vaut
    go = None
    make_subplots = None
    _ABSENCE = str(exc)


def disponible() -> tuple[bool, str]:
    """Plotly est-il installé ? Sinon, la raison, en clair.

    Renvoyer la cause plutôt qu'un simple booléen évite le « ça ne marche
    pas » : l'utilisateur voit qu'il manque une bibliothèque, et laquelle.
    """
    if go is not None:
        return True, ""
    return False, (
        "Le tracé interactif demande la bibliothèque **plotly**, qui n'est pas "
        f"installée ({_ABSENCE}). Installe-la avec `python -m pip install plotly`, "
        "puis relance l'application. L'image fixe, elle, fonctionne sans."
    )


# Boutons de la barre d'outils du tracé. `scrollZoom` autorise la molette ;
# les outils de dessin servent à marquer un front à l'écran avant d'en parler.
CONFIG_MODEBAR: dict = {
    "scrollZoom": True,
    "displaylogo": False,
    "doubleClick": "reset",
    "modeBarButtonsToAdd": ["drawline", "drawrect", "eraseshape"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}

# Ce que fait un glisser sur le tracé.
#
# La sélection ne passe PAS par le bouton habituel de la barre d'outils :
# Plotly le retire de lui-même dès qu'aucune trace ne porte de marqueurs — une
# courbe en trait continu n'a aucun point à sélectionner. Ici la sélection ne
# désigne d'ailleurs pas des points, elle **délimite une plage de temps** que
# l'application relit ensuite à pleine finesse. On l'expose donc comme un mode
# de glisser à part entière, choisi hors du tracé.
GLISSERS: dict[str, str] = {
    "zoom": "Zoomer sur la fenêtre choisie",
    "select": "Relire cette plage à pleine finesse",
}

# Au-delà, le tracé vectoriel devient poussif au zoom : on passe au rendu par
# la carte graphique, qui trace la même chose mais sans peiner.
SEUIL_WEBGL = 12000

# Modes de curseur proposés, du plus lisible au plus précis.
CURSEURS: dict[str, str] = {
    "unifie": "Ligne verticale — toutes les courbes à cet instant",
    "croix": "Croix — la courbe la plus proche du pointeur",
    "aucun": "Aucun",
}


def _rgba(teinte: str, alpha: float) -> str:
    """Teinte hexadécimale → `rgba(...)`.

    Les aplats de zones passent par l'opacité de la forme, mais les vignettes
    de légende sont des marqueurs : eux n'ont pas d'opacité séparée, il faut
    la porter dans la couleur.
    """
    teinte = teinte.lstrip("#")
    r, v, b = (int(teinte[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {v}, {b}, {alpha:.3f})"


def _bornes_avec_zero(valeurs: list[np.ndarray], marge: float = 0.08
                      ) -> tuple[float, float] | None:
    """Bornes verticales englobant zéro, avec une marge.

    Sert au repère « axe Y à zéro » : sans lui, deux couples de 900 et 910 N·m
    paraissent séparés d'un gouffre. Renvoie `None` si aucun point fini n'est
    disponible — auquel cas on laisse Plotly cadrer, plutôt que d'imposer des
    bornes inventées.
    """
    finis = [v[np.isfinite(v)] for v in valeurs]
    finis = [v for v in finis if v.size]
    if not finis:
        return None
    bas = min(0.0, float(min(v.min() for v in finis)))
    haut = max(0.0, float(max(v.max() for v in finis)))
    etendue = (haut - bas) or 1.0
    return bas - marge * etendue, haut + marge * etendue


def lire_au_curseur(t: np.ndarray, courbes: dict[str, np.ndarray], instant: float
                    ) -> tuple[float, dict[str, float]]:
    """Valeurs relevées à l'échantillon le plus proche de `instant`.

    Aucune interpolation : on renvoie l'instant **réellement échantillonné** et
    les valeurs qui y figurent. Interpoler donnerait un nombre qui n'a jamais
    été mesuré, et le curseur servirait alors à lire une valeur inventée.

    Le relevé se fait sur le signal à pleine résolution, pas sur le tracé
    décimé : la valeur affichée est celle du fichier, pas celle du dessin.
    """
    if t.size == 0:
        return float("nan"), {nom: float("nan") for nom in courbes}
    i = int(np.argmin(np.abs(t - instant)))
    return float(t[i]), {
        nom: float(valeurs[i]) if i < valeurs.size else float("nan")
        for nom, valeurs in courbes.items()
    }


def _poser_zones(figure, t, zones, types_zones) -> dict[str, str | None]:
    """Aplats de zones sur tous les panneaux ; renvoie les familles tracées.

    La géométrie vient de `graphiques.zones_visibles` — la même que l'image
    fixe. Les hachures de matplotlib n'ont pas d'équivalent sur une forme
    Plotly : la plage écartée prend un contour pointillé sans remplissage, qui
    porte la même idée — repérée, examinée, non retenue.
    """
    presents: dict[str, str | None] = {}
    for debut, fin, cle in G.zones_visibles(t, zones, types_zones):
        teinte = G.TEINTES_ZONES[cle]
        if teinte is None:
            figure.add_vrect(
                x0=debut, x1=fin, fillcolor="rgba(0,0,0,0)", layer="below",
                line=dict(color=G.TEINTE_DYNAMIQUE, width=1.1, dash="dot"),
            )
        else:
            figure.add_vrect(
                x0=debut, x1=fin, fillcolor=teinte, opacity=G.OPACITE_ZONES[cle],
                layer="below", line_width=0,
            )
        presents.setdefault(cle, teinte)
    return presents


def _vignettes_zones(figure, presents: dict[str, str | None]) -> None:
    """Entrées de légende des zones : un carré par famille tracée.

    Des traces sans point : elles n'existent que pour la légende. Une couleur
    de fond sans libellé ne dirait rien, et la légende de l'image fixe porte
    exactement les mêmes intitulés.
    """
    for cle, teinte in presents.items():
        if teinte is None:  # écartée : carré vide, comme la bande sans aplat
            marqueur = dict(symbol="square", size=11, color="rgba(0,0,0,0)",
                            line=dict(color=G.TEINTE_DYNAMIQUE, width=1.4))
        else:
            marqueur = dict(symbol="square", size=11,
                            color=_rgba(teinte, min(1.0, G.OPACITE_ZONES[cle] * 2.2)),
                            line=dict(width=0))
        figure.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", marker=marqueur,
            name=G.TYPES_ZONES[cle], legendgroup="zones", hoverinfo="skip",
            showlegend=True,
        ), row=1, col=1)


def figure_visualisation(
    t: np.ndarray,
    courbes: dict[str, np.ndarray],
    unites: dict[str, str],
    titre: str = "",
    decimation: int = 1,
    zones=None,
    types_zones: Sequence[str] | None = None,
    curseur: str = "unifie",
    marqueurs: bool = False,
    y_a_zero: bool = False,
    reperes: Sequence[float] = (),
    glisser: str = "zoom",
    hauteur_panneau: int = 260,
):
    """Trace des canaux quelconques en fonction du temps, manipulable à l'écran.

    Même composition que `graphiques.figure_visualisation` : un panneau par
    unité, tous sur le même axe des temps, une couleur par canal tenue d'un
    panneau à l'autre. Superposer un couple en N·m et un régime en tr/min sur
    un axe unique écraserait l'un des deux ; leur donner deux échelles
    verticales inventerait une corrélation que les données ne portent pas.

    Ce que l'interaction ajoute :

    * **zoom** au rectangle, à la molette, et déplacement — l'axe des temps est
      partagé, zoomer un panneau cadre tous les autres au même instant ;
    * `glisser="select"` fait du glisser une **désignation de plage** plutôt
      qu'un zoom : l'appelant relit alors le fichier sur cet intervalle, ce qui
      apporte des points là où le zoom se contente d'agrandir les existants ;
    * **curseur** : `unifie` suit une verticale et liste toutes les courbes à
      cet instant, `croix` pointe la valeur la plus proche, `aucun` désactive ;
    * **légende cliquable** : masquer une courbe sans relancer le tracé ;
    * `marqueurs` matérialise chaque point tracé — au zoom, c'est le seul moyen
      de voir où sont les échantillons et où le tracé n'est qu'une droite ;
    * `reperes` pose des verticales aux instants mesurés (curseurs de mesure).

    `double-clic` remet le cadrage d'origine.
    """
    if go is None:  # pragma: no cover - garde-fou, l'appelant teste `disponible`
        raise RuntimeError(disponible()[1])

    noms = list(courbes)
    couleurs = {nom: G.SERIES[i % len(G.SERIES)] for i, nom in enumerate(noms)}

    groupes: dict[str, list[str]] = {}
    for nom in noms:
        groupes.setdefault(unites.get(nom) or "sans unité", []).append(nom)

    figure = make_subplots(
        rows=len(groupes), cols=1, shared_xaxes=True, vertical_spacing=0.06,
    )
    # Le rendu par la carte graphique ne s'impose qu'aux tracés denses : il est
    # plus fluide, mais son anticrénelage est moins fin sur les courbes courtes.
    trace = go.Scattergl if np.size(t) > SEUIL_WEBGL else go.Scatter

    for rang, (unite, membres) in enumerate(groupes.items(), start=1):
        for nom in membres:
            figure.add_trace(trace(
                x=t, y=courbes[nom], name=nom, mode="lines+markers" if marqueurs
                else "lines",
                line=dict(color=couleurs[nom], width=1.6),
                marker=dict(size=4, color=couleurs[nom]),
                hovertemplate=f"%{{y:.2f}} {unite}<extra>{nom}</extra>",
                legendgroup=nom,
            ), row=rang, col=1)
        figure.update_yaxes(title_text=unite, row=rang, col=1)
        if y_a_zero:
            bornes = _bornes_avec_zero([courbes[nom] for nom in membres])
            if bornes is not None:
                figure.update_yaxes(range=list(bornes), row=rang, col=1)

    presents = _poser_zones(figure, t, zones, types_zones)
    if presents:
        _vignettes_zones(figure, presents)

    for instant in reperes:
        figure.add_vline(
            x=float(instant), line=dict(color=G.ENCRE_2, width=1.1, dash="dash"),
            layer="below",
        )

    _habiller(figure, groupes, titre, decimation, curseur, glisser, hauteur_panneau)
    return figure


def _habiller(figure, groupes: dict, titre: str, decimation: int, curseur: str,
              glisser: str, hauteur_panneau: int) -> None:
    """Chrome récessif, identique à celui de l'image fixe.

    Grille en filet plein, deux montants d'axe, encre de texte pour tout ce qui
    n'est pas une courbe : la couleur reste réservée aux données.
    """
    spikes = dict(
        showspikes=True, spikemode="across", spikesnap="cursor",
        spikecolor=G.ATTENUE, spikethickness=1, spikedash="solid",
    )
    survol = {
        "unifie": dict(hovermode="x unified", hoversubplots="axis"),
        "croix": dict(hovermode="closest"),
        "aucun": dict(hovermode=False),
    }.get(curseur, dict(hovermode="x unified", hoversubplots="axis"))

    figure.update_layout(
        paper_bgcolor=G.FOND, plot_bgcolor=G.FOND,
        font=dict(family="sans-serif", size=12, color=G.ENCRE),
        margin=dict(l=70, r=24, t=76 if titre else 52, b=64),
        height=110 + hauteur_panneau * len(groupes),
        dragmode="select" if glisser == "select" else "zoom",
        selectdirection="h",  # une plage de temps, jamais une bande de valeurs
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left",
                    x=0, font=dict(size=11, color=G.ENCRE_2),
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=G.FOND, bordercolor=G.GRILLE,
                        font=dict(size=11, color=G.ENCRE)),
        **survol,
    )
    if titre:
        figure.update_layout(title=dict(
            text=titre, x=0.0, xanchor="left", y=0.985, yanchor="top",
            font=dict(size=14, color=G.ENCRE),
        ))
    figure.update_xaxes(
        gridcolor=G.GRILLE, gridwidth=1, zeroline=False, showline=True,
        linecolor=G.AXE, ticks="outside", tickcolor=G.AXE,
        tickfont=dict(size=10, color=G.ATTENUE),
        # Trois décimales sous le curseur : c'est la milliseconde qui se lit
        # ici, et deux chiffres l'arrondiraient au-delà de ce qu'on cherche.
        hoverformat=".3f",
        **({} if curseur == "aucun" else spikes),
    )
    figure.update_yaxes(
        gridcolor=G.GRILLE, gridwidth=1, zeroline=False, showline=True,
        linecolor=G.AXE, ticks="outside", tickcolor=G.AXE,
        tickfont=dict(size=10, color=G.ATTENUE),
        title_font=dict(size=11, color=G.ENCRE_2),
        **({} if curseur != "croix" else spikes),
    )
    figure.update_xaxes(title_text="Temps (s)", row=len(groupes), col=1)

    if decimation > 1:
        # La mention est portée par la figure, pas seulement par la page : elle
        # suit l'image exportée, où le zoom n'existe plus pour la démentir.
        figure.add_annotation(
            text=f"Affichage allégé : 1 point sur {decimation}. Le zoom agrandit "
                 "ces points, il n'en rétablit aucun — resserrer la plage de "
                 "temps, si.",
            xref="paper", yref="paper", x=0, y=-0.10 if len(groupes) > 1 else -0.22,
            xanchor="left", yanchor="top", showarrow=False,
            font=dict(size=10, color=G.ATTENUE),
        )
