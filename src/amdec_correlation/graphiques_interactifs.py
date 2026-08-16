"""Tracés interactifs — zoom, déplacement, curseur de lecture.

Contrepartie manipulable des figures de `graphiques`, qui rend des images fixes.
Chaque figure du rapport a ici son équivalent, sur les **mêmes tableaux** et avec
la **même palette** : régression du balayage, résidu thermique, recalage
temporel, répétabilité, redondance des voies, zones détectées, visualisation
libre. Les deux familles coexistent volontairement :

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


def _vignettes_zones(figure, presents: dict[str, str | None],
                     premier_panneau: bool = True) -> None:
    """Entrées de légende des zones : un carré par famille tracée.

    Des traces sans point : elles n'existent que pour la légende. Une couleur
    de fond sans libellé ne dirait rien, et la légende de l'image fixe porte
    exactement les mêmes intitulés.

    `premier_panneau` ne vaut que pour les figures à panneaux : une figure d'un
    seul tenant n'a pas de grille de sous-tracés à référencer.
    """
    cible = dict(row=1, col=1) if premier_panneau else {}
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
        ), **cible)


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
    """Chrome de la visualisation libre : panneaux empilés sur un axe de temps."""
    _chrome(figure, titre=titre, hauteur=110 + hauteur_panneau * len(groupes),
            survol=curseur, legende=True)
    figure.update_layout(
        dragmode="select" if glisser == "select" else "zoom",
        selectdirection="h",  # une plage de temps, jamais une bande de valeurs
    )
    _styler_axes(figure, spikes_x=curseur != "aucun", spikes_y=curseur == "croix",
                 hoverformat_x=".3f")
    figure.update_xaxes(title_text="Temps (s)", row=len(groupes), col=1)

    if decimation > 1:
        # La mention est portée par la figure, pas seulement par la page : elle
        # suit l'image exportée, où le zoom n'existe plus pour la démentir.
        _note(figure, f"Affichage allégé : 1 point sur {decimation}. Le zoom "
                      "agrandit ces points, il n'en rétablit aucun — resserrer "
                      "la plage de temps, si.",
              y=-0.10 if len(groupes) > 1 else -0.22)


# ---------------------------------------------------------------------------
# Chrome commun à toutes les figures
# ---------------------------------------------------------------------------

# Ligne de visée qui suit le pointeur. « across » la fait traverser tout le
# panneau : viser une date ne doit pas demander de viser aussi la courbe.
_SPIKES = dict(showspikes=True, spikemode="across", spikesnap="cursor",
               spikecolor=G.ATTENUE, spikethickness=1, spikedash="solid")

_SURVOLS = {
    "unifie": dict(hovermode="x unified", hoversubplots="axis"),
    "croix": dict(hovermode="closest"),
    "aucun": dict(hovermode=False),
    "proche": dict(hovermode="closest"),
}


def _chrome(figure, titre: str = "", hauteur: int = 440, survol: str = "proche",
            legende: bool = True, marge_gauche: int = 70,
            marge_basse: int = 64) -> None:
    """Habillage récessif commun, identique à celui des images fixes.

    Fond, encre, grille en filet plein, légende en bandeau au-dessus du tracé :
    tout ce qui n'est pas une donnée reste en retrait. La couleur demeure
    réservée aux séries — un lecteur qui apprend l'association sur une figure la
    retrouve sur toutes les autres, fixes comprises.
    """
    figure.update_layout(
        paper_bgcolor=G.FOND, plot_bgcolor=G.FOND,
        font=dict(family="sans-serif", size=12, color=G.ENCRE),
        margin=dict(l=marge_gauche, r=24, t=76 if titre else 52, b=marge_basse),
        height=hauteur,
        showlegend=legende,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left",
                    x=0, font=dict(size=11, color=G.ENCRE_2),
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=G.FOND, bordercolor=G.GRILLE,
                        font=dict(size=11, color=G.ENCRE)),
        **_SURVOLS.get(survol, _SURVOLS["proche"]),
    )
    if titre:
        figure.update_layout(title=dict(
            text=titre, x=0.0, xanchor="left", y=0.985, yanchor="top",
            font=dict(size=14, color=G.ENCRE),
        ))


def _styler_axes(figure, spikes_x: bool = False, spikes_y: bool = False,
                 hoverformat_x: str | None = None) -> None:
    """Grille et montants d'axe, en teinte de chrome et non de donnée."""
    commun = dict(gridcolor=G.GRILLE, gridwidth=1, zeroline=False, showline=True,
                  linecolor=G.AXE, ticks="outside", tickcolor=G.AXE,
                  tickfont=dict(size=10, color=G.ATTENUE))
    figure.update_xaxes(
        **commun, title_font=dict(size=11, color=G.ENCRE_2),
        **({"hoverformat": hoverformat_x} if hoverformat_x else {}),
        **(_SPIKES if spikes_x else {}),
    )
    figure.update_yaxes(
        **commun, title_font=dict(size=11, color=G.ENCRE_2),
        **(_SPIKES if spikes_y else {}),
    )


def _encadre(figure, texte: str, position: str = "haut gauche", **ancrage) -> None:
    """Encadré de valeurs, en encre de texte — jamais en couleur de série.

    Pendant de `graphiques._annoter`. Les chiffres qui commentent une figure
    sont du texte : les peindre de la couleur d'une série les ferait passer
    pour une donnée de cette série.
    """
    coins = {
        "haut gauche": dict(x=0.015, y=0.98, xanchor="left", yanchor="top"),
        "haut droite": dict(x=0.985, y=0.98, xanchor="right", yanchor="top"),
        "bas droite": dict(x=0.985, y=0.03, xanchor="right", yanchor="bottom"),
        "bas gauche": dict(x=0.015, y=0.03, xanchor="left", yanchor="bottom"),
    }[position]
    coins.update(ancrage)
    figure.add_annotation(
        text=texte.replace("\n", "<br>"), xref="paper", yref="paper",
        showarrow=False, align="left", font=dict(size=11, color=G.ENCRE_2),
        bgcolor=G.FOND, bordercolor=G.GRILLE, borderwidth=1, borderpad=6,
        **coins,
    )


def _note(figure, texte: str, y: float = -0.16) -> None:
    """Mention en pied de figure : une précision de lecture, pas une valeur."""
    figure.add_annotation(
        text=texte, xref="paper", yref="paper", x=0, y=y, xanchor="left",
        yanchor="top", showarrow=False, font=dict(size=10, color=G.ATTENUE),
    )


def _points(x, y, couleur: str, nom: str | None, symbole: str = "circle",
            taille: int = 9, gabarit: str | None = None, custom=None,
            **extra):
    """Marqueurs cerclés du fond : deux points qui se recouvrent restent séparés."""
    return go.Scatter(
        x=x, y=y, mode="markers", name=nom or "",
        showlegend=nom is not None,
        marker=dict(color=couleur, size=taille, symbol=symbole,
                    line=dict(color=G.FOND, width=1.4)),
        hovertemplate=gabarit, customdata=custom, **extra,
    )


# ---------------------------------------------------------------------------
# Zones détectées
# ---------------------------------------------------------------------------


def figure_zones(t, reference, mesure, zones, pleine_echelle_Nm: float,
                 titre: str = "", decimation: int = 1,
                 types: Sequence[str] | None = None, curseur: str = "unifie"):
    """Situe sur le signal les zones détectées, chacune identifiée par son type.

    Même contenu que l'image fixe : le tableau de l'onglet dit *combien*, la
    figure dit *où*. Ce que le zoom ajoute est décisif ici — une plage dynamique
    qui déborde de quelques dixièmes de seconde sur un transitoire d'arrêt ne se
    voit pas à l'échelle d'une acquisition de dix minutes.
    """
    if go is None:  # pragma: no cover - l'appelant teste `disponible`
        raise RuntimeError(disponible()[1])

    pas = max(1, int(decimation))
    t = np.asarray(t)[::pas]
    figure = go.Figure()
    trace = go.Scattergl if t.size > SEUIL_WEBGL else go.Scatter
    for valeurs, nom, teinte in (
        (np.asarray(reference)[::pas], "couple de référence banc", G.SERIE_1),
        (np.asarray(mesure)[::pas], "couple mesuré transmissions", G.SERIE_2),
    ):
        figure.add_trace(trace(
            x=t, y=valeurs, name=nom, mode="lines",
            line=dict(color=teinte, width=1.4),
            hovertemplate="%{y:.1f} N·m<extra>" + nom + "</extra>",
        ))

    presents = _poser_zones(figure, t, zones, types)
    if presents:
        _vignettes_zones(figure, presents, premier_panneau=False)

    _chrome(figure, titre=titre or getattr(zones, "chemin", None) and zones.chemin.name,
            hauteur=470, survol=curseur)
    _styler_axes(figure, spikes_x=curseur != "aucun", spikes_y=curseur == "croix",
                 hoverformat_x=".3f")
    figure.update_xaxes(title_text="Temps (s)")
    figure.update_yaxes(title_text="Couple (N·m)")

    alimente = ", ".join(zones.contributions()) or "aucune grandeur"
    note = f"{zones.profil()}  ·  alimente : {alimente}"
    if getattr(zones, "paliers", None):
        # Sans cette précision, la bande plus étroite que le plateau se lit comme
        # une détection incomplète, alors que c'est le contraire : seule la part
        # établie est moyennée.
        note += ("  ·  les bandes de palier montrent la part effectivement "
                 "moyennée, pas toute la plage stable")
    _note(figure, note, y=-0.20)
    if pas > 1:
        _note(figure, f"Affichage allégé : 1 point sur {pas}. Le zoom agrandit "
                      "ces points, il n'en rétablit aucun.", y=-0.27)
    return figure


# ---------------------------------------------------------------------------
# Régression du balayage
# ---------------------------------------------------------------------------


def _detail_palier(paliers) -> tuple[list, str]:
    """Colonnes de survol d'un palier : quand, à quelle température, quel résidu.

    Le nuage de la régression perd le temps : deux points superposés peuvent
    venir d'instants — et de températures — très différents. Le survol le rend,
    ce que l'image fixe ne peut pas faire.
    """
    custom = [
        [p.t_debut, p.t_fin, p.residu,
         float("nan") if p.temperature is None else p.temperature]
        for p in paliers
    ]
    gabarit = (
        "référence %{x:.1f} N·m<br>mesure %{y:.1f} N·m<br>"
        "résidu %{customdata[2]:+.2f} N·m<br>"
        "de %{customdata[0]:.1f} à %{customdata[1]:.1f} s<br>"
        "%{customdata[3]:.1f} °C"
    )
    return custom, gabarit


def figure_regression_balayage(paliers, reg, hyst, pleine_echelle_Nm: float,
                               titre: str = "Balayage couple — corrélation "
                                            "transmissions / banc GMP"):
    """Régression des points stabilisés et résidus, en deux panneaux liés.

    Survoler un point donne l'instant du palier et sa température : c'est ce
    qu'il faut pour comprendre un résidu qui sort du lot, et c'est précisément
    ce qu'un nuage figé ne peut pas dire.
    """
    if go is None:  # pragma: no cover
        raise RuntimeError(disponible()[1])

    figure = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           row_heights=[0.7, 0.3], vertical_spacing=0.07)

    if not reg.non_calculable and paliers:
        bornes = [min(p.reference for p in paliers), max(p.reference for p in paliers)]
        xs = np.linspace(bornes[0], bornes[1], 200)
        figure.add_trace(go.Scatter(
            x=xs, y=reg.a * xs + reg.b, mode="lines", name="régression linéaire",
            line=dict(color=G.ATTENUE, width=1.4), hoverinfo="skip",
        ), row=1, col=1)

    for sens, libelle, teinte, symbole in (
        ("montee", "montée", G.SERIE_1, "circle"),
        ("descente", "descente", G.SERIE_2, "triangle-up"),
    ):
        lot = [p for p in paliers if p.sens == sens]
        if not lot:
            continue
        custom, gabarit = _detail_palier(lot)
        figure.add_trace(_points(
            [p.reference for p in lot], [p.mesure for p in lot], teinte, libelle,
            symbole=symbole, custom=custom,
            gabarit=gabarit + f"<extra>{libelle}</extra>",
        ), row=1, col=1)

    if not reg.non_calculable:
        residus_Nm = reg.residus
        sens_tous = [p.sens for p in paliers][: residus_Nm.size]
        figure.add_hline(y=0, line=dict(color=G.AXE, width=1), row=2, col=1)
        for sens, teinte, symbole in (("montee", G.SERIE_1, "circle"),
                                      ("descente", G.SERIE_2, "triangle-up")):
            idx = [i for i, s in enumerate(sens_tous) if s == sens]
            if not idx:
                continue
            figure.add_trace(_points(
                [reg.x[i] for i in idx], [residus_Nm[i] for i in idx], teinte, None,
                symbole=symbole,
                gabarit="référence %{x:.1f} N·m<br>résidu %{y:+.2f} N·m<extra></extra>",
            ), row=2, col=1)
        marge = max(float(np.max(np.abs(residus_Nm))) * 1.4, 0.05)
        figure.update_yaxes(range=[-marge, marge], row=2, col=1)

        lignes = [
            f"a = {reg.a:.4f} ± {reg.sigma_a:.4f}",
            f"b = {reg.b:+.2f} ± {reg.sigma_b:.2f} N·m",
            f"R² = {reg.r2:.5f}   (n = {reg.n} paliers)",
        ]
        if not hyst.non_calculable:
            lignes.append(f"hystérésis max = {hyst.max_Nm:.2f} N·m")
        _encadre(figure, "\n".join(lignes))
    else:
        _encadre(figure, reg.non_calculable.replace(" : ", " :\n"))

    _chrome(figure, titre=titre, hauteur=620)
    _styler_axes(figure)
    figure.update_yaxes(title_text="Couple mesuré transmissions (N·m)", row=1, col=1)
    figure.update_yaxes(title_text="Résidu (N·m)", row=2, col=1)
    figure.update_xaxes(title_text="Couple de référence banc GMP (N·m)", row=2, col=1)
    return figure


# ---------------------------------------------------------------------------
# Résidu en fonction de la température
# ---------------------------------------------------------------------------


def figure_residu_temperature(sens_th, pleine_echelle_Nm: float,
                              titre: str = "Résidu (mesuré − référence) en "
                                           "fonction de la température"):
    """Nuage des échantillons, moyennes par classe, et droite de sensibilité.

    Le cadrage sur les centiles 1–99 de l'image fixe n'est plus nécessaire : ici
    l'axe part large et c'est le lecteur qui zoome. Aucun point n'est écarté —
    ni du calcul, ni du tracé.
    """
    if go is None:  # pragma: no cover
        raise RuntimeError(disponible()[1])

    figure = go.Figure()
    if sens_th.nuage_T.size:
        pas = max(1, sens_th.nuage_T.size // 6000)
        figure.add_trace(go.Scattergl(
            x=sens_th.nuage_T[::pas],
            y=sens_th.nuage_res[::pas],
            mode="markers", name="échantillons",
            marker=dict(color=G.BLEU_CLAIR, size=3.5, opacity=0.45),
            hovertemplate="%{x:.1f} °C<br>%{y:+.2f} N·m<extra>échantillon</extra>",
        ))
    if sens_th.temperatures.size:
        figure.add_trace(_points(
            sens_th.temperatures, sens_th.residus,
            G.SERIE_1, "moyenne par classe de température",
            gabarit="%{x:.1f} °C<br>%{y:+.2f} N·m<extra>classe</extra>",
        ))
    if not sens_th.non_calculable and sens_th.temperatures.size:
        xs = np.linspace(sens_th.temperatures.min(), sens_th.temperatures.max(), 100)
        # La droite est reconstruite à partir de la pente et du barycentre des classes.
        y0 = float(np.mean(sens_th.residus))
        x0 = float(np.mean(sens_th.temperatures))
        figure.add_trace(go.Scatter(
            x=xs, y=y0 + sens_th.Nm_par_C * (xs - x0), mode="lines",
            name="régression linéaire", line=dict(color=G.ATTENUE, width=1.4),
            hoverinfo="skip",
        ))
        lignes = [
            f"sensibilité = {sens_th.Nm_par_C:+.4f} N·m/°C",
            f"soit {sens_th.Nm_pour_10C:+.2f} N·m pour 10 °C",
            f"R² = {sens_th.r2:.3f}   ({sens_th.n_classes} cellules)",
            f"excursion = {sens_th.amplitude_C:.1f} °C",
        ]
        if sens_th.correction_couple:
            lignes.append("régression multiple, effet du couple retiré\n"
                          f"(coefficient {sens_th.coefficient_couple:+.4f} N·m/N·m)")
        _encadre(figure, "\n".join(lignes))
    elif sens_th.non_calculable:
        _encadre(figure, sens_th.non_calculable.replace(" : ", " :\n"))

    figure.add_hline(y=0, line=dict(color=G.AXE, width=1))
    _chrome(figure, titre=titre, hauteur=470)
    _styler_axes(figure)
    figure.update_xaxes(title_text="Température (°C)")
    figure.update_yaxes(title_text=(
        "Résidu corrigé de l'effet couple (N·m)" if sens_th.correction_couple
        else "Résidu (N·m)"
    ))
    return figure


# ---------------------------------------------------------------------------
# Recalage temporel
# ---------------------------------------------------------------------------


def figure_recalage(t, reference, mesure, recal, titre: str = "Recalage temporel",
                    duree_zoom_s: float = 20.0, curseur: str = "unifie",
                    facteur_charge: float = 10.0):
    """Avant recalage, après recalage, et pic d'intercorrélation.

    L'image fixe doit choisir une fenêtre de zoom, faute de quoi un décalage de
    quelques dizaines de millisecondes serait invisible à l'échelle du cycle.
    Ici le cadrage initial reste le même, mais il n'enferme plus : `facteur_charge`
    fois cette durée est envoyée **à pleine résolution**, et un double-clic la
    déplie. C'est le seul moyen de vérifier qu'un recalage juste sur la fenêtre
    corrélée l'est aussi de part et d'autre.

    Pourquoi ne pas tout envoyer : un décalage de 40 ms vaut quatre échantillons
    à 100 Hz. Décimer pour alléger effacerait exactement ce que la figure sert à
    lire, et envoyer une acquisition entière sans décimer alourdirait la page
    d'un facteur cent. On envoie donc un voisinage large, intact.
    """
    if go is None:  # pragma: no cover
        raise RuntimeError(disponible()[1])

    t = np.asarray(t, dtype=float)
    reference = np.asarray(reference)
    mesure = np.asarray(mesure)

    centre = (0.5 * (recal.fenetre[0] + recal.fenetre[1]) if not recal.non_calculable
              else 0.5 * (t[0] + t[-1]))
    demi_charge = 0.5 * facteur_charge * duree_zoom_s
    charge = (t >= centre - demi_charge) & (t <= centre + demi_charge)
    if charge.sum() < 2:
        charge = np.ones(t.size, dtype=bool)
    t_charge = t[charge]
    reference, mesure = reference[charge], mesure[charge]

    figure = make_subplots(
        rows=3, cols=1, shared_xaxes=False, vertical_spacing=0.11,
        row_heights=[0.35, 0.35, 0.30],
        subplot_titles=("avant recalage",
                        "après recalage" if not recal.non_calculable
                        else "recalage non appliqué",
                        "fonction d'intercorrélation"),
    )
    trace = go.Scattergl if t_charge.size > SEUIL_WEBGL else go.Scatter

    def _panneau(rang: int, y_mesure, premier: bool) -> None:
        for valeurs, nom, teinte in (
            (reference, "référence banc GMP", G.SERIE_1),
            (y_mesure, "couple mesuré transmissions", G.SERIE_2),
        ):
            figure.add_trace(trace(
                x=t_charge, y=valeurs, name=nom, mode="lines",
                line=dict(color=teinte, width=1.5), legendgroup=nom,
                showlegend=premier,
                hovertemplate="%{y:.1f} N·m<extra>" + nom + "</extra>",
            ), row=rang, col=1)

    _panneau(1, mesure, premier=True)
    if not recal.non_calculable:
        # Le retard est appliqué sur le voisinage chargé, avec sa propre base de
        # temps : décaler après découpe donnerait des bords artificiels.
        _panneau(2, G.appliquer_retard(t_charge, mesure, recal.retard_ms / 1000.0),
                 False)
    else:
        _panneau(2, mesure, False)

    # Cadrage initial : la fenêtre effectivement corrélée. Le voisinage chargé
    # est dix fois plus large, à un double-clic près.
    debut = max(float(t_charge[0]), centre - duree_zoom_s / 2)
    fin = min(float(t_charge[-1]), debut + duree_zoom_s)
    for rang in (1, 2):
        figure.update_xaxes(range=[debut, fin], row=rang, col=1)

    if not recal.non_calculable:
        figure.add_trace(go.Scatter(
            x=recal.retards_ms, y=recal.correlation, mode="lines",
            name="intercorrélation normalisée", line=dict(color=G.SERIE_1, width=1.5),
            hovertemplate="%{x:+.1f} ms<br>r = %{y:.3f}<extra></extra>",
        ), row=3, col=1)
        figure.add_vline(
            x=recal.retard_ms, line=dict(color=G.SERIE_2, width=1.5),
            row=3, col=1,
            annotation_text=f"pic : {recal.retard_ms:+.1f} ms",
            annotation_position="top", annotation_font=dict(size=11, color=G.ENCRE_2),
        )
        _encadre(figure, f"RMS résidu : {recal.rms_residu_avant_Nm:.1f} → "
                         f"{recal.rms_residu_apres_Nm:.1f} N·m  "
                         f"({100 * recal.gain_rms:+.0f} %)",
                 position="haut droite", y=0.62)
    else:
        _encadre(figure, recal.non_calculable.replace(" : ", " :\n"),
                 position="haut droite", y=0.62)

    _chrome(figure, titre=titre, hauteur=820, survol=curseur)
    _styler_axes(figure, spikes_x=curseur != "aucun", hoverformat_x=".3f")
    for rang in (1, 2):
        figure.update_yaxes(title_text="Couple (N·m)", row=rang, col=1)
        figure.update_xaxes(title_text="Temps (s)", row=rang, col=1)
    figure.update_xaxes(title_text="Décalage appliqué à la mesure (ms)",
                        row=3, col=1, hoverformat=".1f")
    figure.update_yaxes(title_text="r (−)", row=3, col=1)
    for annotation in figure.layout.annotations[:3]:
        annotation.update(font=dict(size=12, color=G.ENCRE), x=0, xanchor="left")
    _note(figure, f"Cadré d'emblée sur la fenêtre corrélée ; un double-clic déplie "
                  f"les {t_charge[-1] - t_charge[0]:.0f} s chargées autour d'elle, "
                  "à pleine résolution.", y=-0.09)
    return figure


# ---------------------------------------------------------------------------
# Répétabilité
# ---------------------------------------------------------------------------


def figure_repetabilite(rep, pleine_echelle_Nm: float,
                        titre: str = "Répétabilité — dispersion du résidu à "
                                     "couple de référence constant"):
    """±2σ par niveau de couple, et la bande poolée qui les résume."""
    if go is None:  # pragma: no cover
        raise RuntimeError(disponible()[1])

    figure = go.Figure()
    if rep.non_calculable or not rep.groupes:
        _encadre(figure, (rep.non_calculable or "aucun groupe exploitable")
                 .replace(" : ", " :\n"))
        _chrome(figure, titre=titre, hauteur=420, legende=False)
        _styler_axes(figure)
        figure.update_xaxes(title_text="Couple de référence banc GMP (N·m)")
        figure.update_yaxes(title_text="Résidu (N·m)")
        return figure

    x = [g.reference_moyenne for g in rep.groupes]
    s = [g.ecart_type_residu_Nm for g in rep.groupes]
    figure.add_hline(y=0, line=dict(color=G.AXE, width=1))
    figure.add_trace(go.Scatter(
        x=x, y=[0.0] * len(x), mode="markers", name="±2σ par niveau de couple",
        marker=dict(color=G.SERIE_1, size=9,
                    line=dict(color=G.FOND, width=1.4)),
        error_y=dict(type="data", array=[2 * v for v in s], color=G.SERIE_1,
                     thickness=1.4, width=5),
        customdata=[[2 * v, g.n] for v, g in zip(s, rep.groupes)],
        hovertemplate="référence %{x:.1f} N·m<br>±2σ = %{customdata[0]:.2f} N·m"
                      "<br>%{customdata[1]} paliers<extra></extra>",
    ))
    for signe in (1, -1):
        figure.add_hline(
            y=signe * 2 * rep.ecart_type_Nm,
            line=dict(color=G.ATTENUE, width=1.2),
        )
    figure.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines", line=dict(color=G.ATTENUE, width=1.2),
        name=f"±2σ poolé = ±{2 * rep.ecart_type_Nm:.2f} N·m", hoverinfo="skip",
    ))
    _encadre(figure,
             f"σ poolé = {rep.ecart_type_Nm:.3f} N·m\n"
             f"{rep.n_points} paliers, {len(rep.groupes)} niveaux, "
             f"{rep.degres_liberte} ddl")
    _chrome(figure, titre=titre, hauteur=450)
    _styler_axes(figure)
    figure.update_xaxes(title_text="Couple de référence banc GMP (N·m)")
    figure.update_yaxes(title_text="Résidu (N·m)")
    return figure


# ---------------------------------------------------------------------------
# Redondance gauche / droite
# ---------------------------------------------------------------------------


def figure_redondance(t, gauche, droite, pleine_echelle_Nm: float,
                      titre: str = "Redondance des voies — résidu gauche − droite",
                      decimation: int = 1, curseur: str = "unifie"):
    """Les deux voies, et leur écart en N·m sous elles."""
    if go is None:  # pragma: no cover
        raise RuntimeError(disponible()[1])

    pas = max(1, int(decimation))
    t = np.asarray(t)[::pas]
    gauche = np.asarray(gauche)[::pas]
    droite = np.asarray(droite)[::pas]

    figure = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           row_heights=[0.6, 0.4], vertical_spacing=0.07)
    trace = go.Scattergl if t.size > SEUIL_WEBGL else go.Scatter
    for valeurs, nom, teinte in ((gauche, "transmission gauche", G.SERIE_1),
                                 (droite, "transmission droite", G.SERIE_2)):
        figure.add_trace(trace(
            x=t, y=valeurs, name=nom, mode="lines",
            line=dict(color=teinte, width=1.4),
            hovertemplate="%{y:.1f} N·m<extra>" + nom + "</extra>",
        ), row=1, col=1)

    ecart = gauche - droite
    figure.add_hline(y=0, line=dict(color=G.AXE, width=1), row=2, col=1)
    figure.add_trace(trace(
        x=t, y=ecart, name="gauche − droite", mode="lines",
        line=dict(color=G.SERIE_3, width=1.2),
        hovertemplate="%{y:+.2f} N·m<extra>gauche − droite</extra>",
    ), row=2, col=1)

    fini = ecart[np.isfinite(ecart)]
    if fini.size:
        _encadre(figure, f"moyenne {fini.mean():+.3f} N·m — "
                         f"σ {fini.std(ddof=1):.3f} N·m",
                 position="bas droite")
    _chrome(figure, titre=titre, hauteur=580, survol=curseur)
    _styler_axes(figure, spikes_x=curseur != "aucun", hoverformat_x=".3f")
    figure.update_yaxes(title_text="Couple (N·m)", row=1, col=1)
    figure.update_yaxes(title_text="Écart (N·m)", row=2, col=1)
    figure.update_xaxes(title_text="Temps (s)", row=2, col=1)
    if pas > 1:
        _note(figure, f"Affichage allégé : 1 point sur {pas}.", y=-0.13)
    return figure
