"""Interface graphique Streamlit du pipeline de corrélation couple.

Cette page n'implémente aucun traitement : elle assemble une configuration,
appelle la bibliothèque et affiche ce qui en sort. Les chiffres à l'écran sont
donc les mêmes que ceux du rapport, et la configuration exportée rejoue à
l'identique en ligne de commande.

GESTION DE L'ÉTAT — Streamlit réexécute tout le script à chaque interaction.
Chaque widget porte donc une **clé stable** et `st.session_state` est l'unique
source de vérité : aucune valeur n'est recopiée à la main d'un côté à l'autre,
et aucun défaut n'est recalculé à chaque passage. Sans cette discipline, la
moindre saisie réinitialiserait les widgets rendus avant elle.

Lancement :  python scripts/03_interface.py
"""

from __future__ import annotations

import io
import math
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

_RACINE_PROJET = Path(__file__).resolve().parents[3]
if str(_RACINE_PROJET / "src") not in sys.path:
    sys.path.insert(0, str(_RACINE_PROJET / "src"))

from amdec_correlation import analyse as A  # noqa: E402
from amdec_correlation import graphiques  # noqa: E402
from amdec_correlation import inventaire as I  # noqa: E402
from amdec_correlation import lecteurs  # noqa: E402
from amdec_correlation import rapport as R  # noqa: E402
from amdec_correlation import zones as Zn  # noqa: E402
from amdec_correlation.config import MODES_COMPARAISON  # noqa: E402
from amdec_correlation.interface import etat as E  # noqa: E402
from amdec_correlation.interface import explorateur  # noqa: E402

# Palette partagée avec les figures : l'écran et les PNG se lisent comme un seul
# système visuel.
ENCRE_2 = "#52514e"
ATTENUE = "#898781"
BON = "#0ca30c"
ATTENTION = "#fab219"
CRITIQUE = "#d03b3b"

FICHIER_CONFIG = _RACINE_PROJET / "config" / "correlation.yaml"

# Mémoire de la dernière utilisation. Rangée dans le profil de l'utilisateur, et
# non dans le projet : elle doit survivre à une mise à jour du dépôt, et deux
# comptes du même poste ne doivent pas se marcher dessus. Le mapping des canaux
# est stable d'une campagne à l'autre — le redéclarer à chaque lancement est la
# corvée que cette mémoire supprime.
FICHIER_MEMOIRE = Path.home() / ".amdec_correlation" / "derniere_configuration.yaml"

st.set_page_config(
    page_title="Corrélation couple — banc GMP",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1400px; }
      hr { border: none; border-top: 1px solid #e1e0d9; margin: 1.2rem 0; }
      div[data-testid="stMetricValue"] { font-size: 1.55rem; }
      .etape { color: #52514e; font-size: 0.86rem; line-height: 1.7; }
      .aide { color: #898781; font-size: 0.84rem; line-height: 1.55; }
      table { font-size: 0.88rem; }
      code { font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# État
# ---------------------------------------------------------------------------

# Valeurs initiales de tous les widgets scalaires. Elles ne sont posées qu'UNE
# fois : ensuite, les widgets écrivent directement dans `st.session_state`.
DEFAUTS: dict[str, object] = {
    "racine": "", "dossier_sortie": "sortie",
    "pe": 1500.0, "mode": "moyenne", "rapport": 1.0, "freq": 100.0,
    "n_fichiers": 2, "mapping_commun": True,
    "u_ref_connue": False, "u_ref": 2.0,
    "plage_connue": False, "plage": 40.0,
    "remontage": "Non précisé", "ligne_droite": False,
    "p::duree_palier_s": 2.0, "p::tolerance_stab_pc_pe": 0.5,
    "p::fraction_finale": 0.5, "p::ecart_min_paliers_pc_pe": 1.0,
    "p::tolerance_appariement_pc_pe": 1.0,
    "i::retard_max_ms": 500.0, "i::passe_haut_Hz": 0.2,
    "i::seuil_activite_pc_pe": 2.0, "i::fenetre_activite_s": 1.0,
    "i::duree_comblement_s": 5.0,
    "i::n_blocs_coherence": 4, "i::accord_blocs_max_ms": 20.0,
    "i::fraction_score_min": 0.25, "i::duree_min_fenetre_s": 0.0,
    "z::duree_fenetre_s": 5.0, "z::seuil_couple_ref_pc_pe": 1.0,
    "z::seuil_regime": 20.0, "z::fraction_bord": 0.25,
    "t::amplitude_min_C": 5.0,
    "d::seuil_offset_pc_pe": 0.5, "d::seuil_gain_pc": 1.0,
    "d::seuil_retard_ms": 5.0, "d::gain_rms_recalage": 0.30,
    "d::seuil_correlation_point_fct": 0.5,
}

for _cle, _valeur in DEFAUTS.items():
    st.session_state.setdefault(_cle, _valeur)
st.session_state.setdefault("inventaire", {})
st.session_state.setdefault("canaux_communs", {})
st.session_state.setdefault("canaux", {})
st.session_state.setdefault("campagne", None)
st.session_state.setdefault("zones", [])
st.session_state.setdefault("signaux_zones", {})

# Un dossier retenu dans le navigateur est appliqué ICI, avant que le champ de
# saisie correspondant n'existe : Streamlit refuse qu'on modifie la clé d'un
# widget déjà instancié, et le clic sur « Choisir ce dossier » a nécessairement
# lieu après le rendu du champ. La valeur transite donc par une clé en attente.
for _champ in ("racine", "dossier_sortie"):
    _choisi = st.session_state.pop(f"choisi::{_champ}", None)
    if _choisi is not None:
        st.session_state[_champ] = _choisi


def _section(prefixe: str) -> dict[str, object]:
    """Rassemble les paramètres d'une section depuis l'état des widgets."""
    n = len(prefixe) + 2
    return {
        cle[n:]: st.session_state[cle]
        for cle in DEFAUTS
        if cle.startswith(f"{prefixe}::")
    }


def _selectbox_canal(libelle: str, cle: str, options: list[str]) -> None:
    """Liste déroulante de canal, dont l'état applicatif est tenu explicitement.

    Deux pièges, et le second est coûteux :

    1. Si la valeur mémorisée n'existe plus (nouvel inventaire, autre dossier),
       on retombe sur « absent » plutôt que de laisser Streamlit lever.
    2. Les listes de l'onglet Canaux n'apparaissent qu'une fois l'inventaire
       fait, donc lors d'un rerun **ultérieur** à celui qui a posé la valeur —
       c'est le cas de la mémoire, restaurée au tout premier passage. Or un
       widget rendu pour la première fois n'adopte pas une clé pré-alimentée :
       il s'initialise sur son propre défaut, « absent », et l'écrase au passage
       suivant. Le mapping était ainsi **silencieusement perdu**, et l'analyse
       refusait tous les fichiers faute de canal.

    D'où la même discipline que `_case` : le widget porte une clé distincte,
    reçoit sa valeur par `index`, et la recopie dans l'état applicatif.
    """
    courant = st.session_state.get(cle, E.CANAL_ABSENT)
    if courant not in options:
        courant = E.CANAL_ABSENT
    choix = st.selectbox(
        libelle, options, index=options.index(courant), key=f"w::{cle}"
    )
    st.session_state[cle] = choix


def _multiselect_etat(libelle: str, cle: str, options: list[str]) -> None:
    """Canaux d'état, même discipline que `_selectbox_canal` (cf. son docstring)."""
    courant = [v for v in st.session_state.get(cle, []) if v in options]
    choix = st.multiselect(libelle, options, default=courant, key=f"w::{cle}")
    st.session_state[cle] = list(choix)


def _case(libelle: str, cle: str, defaut: bool, aide: str = "") -> bool:
    """Case à cocher dont l'état applicatif est tenu explicitement.

    Les onglets 2 et 3 ne sont rendus qu'une fois l'inventaire fait : leurs
    widgets apparaissent donc pour la première fois lors d'un rerun ultérieur.
    Dans ce cas, une clé pré-alimentée dans `session_state` n'est pas reprise
    par le widget — la case garde son propre état, décorrélé de celui de
    l'application, et les clics restent sans effet. On donne donc au widget une
    clé distincte et on recopie sa valeur dans l'état applicatif.
    """
    valeur = st.checkbox(
        libelle,
        value=bool(st.session_state.get(cle, defaut)),
        key=f"w::{cle}",
        help=aide or None,
    )
    st.session_state[cle] = valeur
    return valeur


def _choisir_dossier(cle: str, libelle: str) -> None:
    """Bouton unique : ouvre l'explorateur du poste et retient le dossier choisi.

    Streamlit s'affiche dans un navigateur web, qui n'a pas le droit d'ouvrir
    l'explorateur du système. C'est donc le processus Python — qui, lui, tourne
    sur le poste — qui l'ouvre, dans un sous-processus (cf. `explorateur`).

    Le chemin retenu n'est pas écrit directement dans l'état du champ de saisie :
    Streamlit interdit de modifier la clé d'un widget déjà instancié dans la même
    exécution, et le clic a nécessairement lieu après le rendu du champ. La
    valeur transite par une clé en attente, appliquée en tête de l'exécution
    suivante.
    """
    ouvrable, motif = explorateur.disponible()
    if st.button(
        "Parcourir…", key=f"{cle}::parcourir", use_container_width=True,
        disabled=not ouvrable,
        help=(f"Ouvre l'explorateur de fichiers du poste pour choisir le dossier "
              f"{libelle}." if ouvrable else f"Explorateur indisponible : {motif}"),
    ):
        try:
            with st.spinner("Fenêtre de sélection ouverte sur le bureau…"):
                choisi = explorateur.choisir_dossier(
                    st.session_state.get(cle, ""),
                    titre=f"Corrélation couple — dossier {libelle}",
                )
        except explorateur.ExplorateurIndisponible as exc:
            st.error(f"Explorateur indisponible : {exc}. Saisis le chemin à la main.")
        else:
            if choisi:
                st.session_state[f"choisi::{cle}"] = choisi
                st.rerun()
    if not ouvrable:
        # Le motif complet est dans l'infobulle du bouton : répété ici en clair,
        # il occuperait à lui seul la moitié de la barre latérale.
        st.caption("Explorateur indisponible — saisis le chemin à la main.")


def _pastille(ok: bool | None, texte: str) -> str:
    """Indicateur d'étape : icône + libellé, jamais la couleur seule."""
    if ok is True:
        return f"<span style='color:{BON}'>●</span> {texte}"
    if ok is False:
        return f"<span style='color:{CRITIQUE}'>●</span> {texte}"
    return f"<span style='color:{ATTENUE}'>○</span> {texte}"


@st.cache_data(show_spinner=False)
def inventorier(racine: str, n_fichiers: int) -> dict[str, dict]:
    """Inventaire des canaux, mis en cache (relecture disque coûteuse)."""
    resultats: dict[str, dict] = {}
    for inv in I.inventorier(Path(racine), None, n_fichiers):
        resultats[inv.dossier] = {
            "n_fichiers": inv.n_fichiers_total,
            "fichiers": [f.name for f in inv.fichiers_examines],
            "coherent": inv.libelles_coherents,
            "erreurs": inv.erreurs,
            "noms": inv.noms_uniques,
            "canaux": [
                {
                    "Canal": d.nom,
                    "Unité": d.unite or "—",
                    "Groupe": d.groupe,
                    "Échantillons": d.n_echantillons,
                    "Fréquence (Hz)": round(d.frequence_Hz, 2) if d.frequence_Hz else None,
                    "t début (s)": round(d.t_debut, 3) if d.t_debut is not None else None,
                    "t fin (s)": round(d.t_fin, 3) if d.t_fin is not None else None,
                }
                for descriptions in inv.canaux.values()
                for d in descriptions
            ],
            "n_groupes": len({d.groupe for desc in inv.canaux.values() for d in desc}),
        }
    return resultats


def _valeur(x: float, decimales: int = 3, signe: bool = False, suffixe: str = "") -> str:
    if x is None or not math.isfinite(x):
        return "non calculable"
    gabarit = f"{{:+.{decimales}f}}" if signe else f"{{:.{decimales}f}}"
    return gabarit.format(x) + suffixe


# ---------------------------------------------------------------------------
# Visualisation libre
# ---------------------------------------------------------------------------

# Au-delà, le tracé s'alourdit sans rien montrer de plus : deux points ne
# peuvent pas occuper le même pixel.
MAX_POINTS_TRACES = 5000

# Chaque combinaison consomme un emplacement de couleur : au-delà, le tracé
# n'est plus lisible avant même d'ajouter un canal brut.
MAX_DERIVEES = 4


@st.cache_data(show_spinner=False)
def fichiers_du_dossier(racine: str, dossier: str) -> list[Path]:
    """Acquisitions d'un groupe de l'inventaire.

    `(racine)` n'est pas un sous-dossier : c'est le groupe des acquisitions
    posées directement dans le dossier racine — le cas d'un dossier plat, qui
    est l'organisation la plus courante. Les concaténer au chemin donnerait
    `racine/(racine)`, qui n'existe pas, et la visualisation ne verrait aucun
    fichier là où il y en a.
    """
    racine = Path(racine)
    if dossier == I.NOM_RACINE:
        return [f for f in lecteurs.lister_fichiers(racine) if f.parent == racine]
    return lecteurs.lister_fichiers(racine / dossier)


@st.cache_data(show_spinner=False)
def canaux_du_fichier(chemin: str) -> dict[str, str]:
    """Nom de canal → unité, pour un fichier précis.

    On interroge le fichier sélectionné, et non l'échantillon de l'inventaire :
    rien ne garantit que tous les fichiers d'un dossier portent les mêmes canaux.

    Les canaux maîtres sont écartés : porter le temps en ordonnée d'un tracé en
    fonction du temps n'apprend rien, et ils se retrouveraient proposés par
    défaut puisqu'ils ouvrent la liste de chaque groupe. Le critère est double
    — nom de temps ET unité en secondes — pour ne pas écarter par erreur un
    canal physique dont le nom commencerait par « t ».
    """
    return {
        d.nom: d.unite
        for d in lecteurs.decrire_canaux(chemin)
        if not (d.nom.strip().lower() in ("t", "time", "temps", "timestamp")
                and d.unite.strip().lower() in ("s", "sec", "second", "secondes"))
    }


def detecter_zones(cfg) -> tuple[list, dict[str, tuple]]:
    """Détecte les zones de chaque acquisition, sans lancer l'analyse complète.

    Permet de vérifier ce que l'outil a trouvé AVANT de calculer quoi que ce
    soit — c'est ce qui rend le mode automatique auditable.

    Renvoie aussi, par acquisition, les signaux **décimés** nécessaires au tracé.
    Les garder plutôt que de pré-calculer une image permet de re-tracer à la
    demande quand l'utilisateur change les familles de zones affichées ; décimés
    à `MAX_POINTS_TRACES`, ils coûtent une centaine de kilo-octets par fichier —
    deux points ne peuvent de toute façon pas occuper le même pixel.
    """
    resultats, signaux = [], {}
    mapping = cfg.canaux_pour("")
    for fichier in lecteurs.lister_fichiers(cfg.racine):
        try:
            donnees = A.preparer(fichier, cfg, mapping)
        except Exception as exc:
            resultats.append(Zn.ZonesFichier(chemin=fichier, duree_s=0.0, erreur=str(exc)))
            continue
        zones = Zn.detecter(donnees, cfg)
        resultats.append(zones)
        pas = max(1, int(np.ceil(donnees.t.size / MAX_POINTS_TRACES)))
        # Les zones sont repérées en secondes : le facteur de décimation n'a
        # pas à être conservé, les aplats retombent au bon endroit quel que
        # soit l'allègement du tracé.
        signaux[fichier.name] = (
            donnees.t[::pas].copy(),
            donnees.reference[::pas].copy(),
            donnees.mesure[::pas].copy(),
        )
    return resultats, signaux


def figure_zones_png(nom: str, zones, signaux: dict, pe: float,
                     types: list[str] | None = None) -> bytes | None:
    """Rend la figure des zones d'une acquisition, selon le filtre courant."""
    if nom not in signaux:
        return None
    t, reference, mesure = signaux[nom]
    figure = graphiques.figure_zones(
        t, reference, mesure, zones, pe,
        titre=f"{nom} — zones détectées", types=types,
    )
    tampon = io.BytesIO()
    figure.savefig(tampon, format="png", dpi=110, bbox_inches="tight")
    plt.close(figure)
    return tampon.getvalue()


def tracer_visualisation(
    fichier: Path, noms: list[str], catalogue: dict[str, str],
    derivees: list[tuple[str, str, str]],
    zones=None, types_zones: list[str] | None = None,
) -> None:
    """Charge les canaux demandés et les trace, groupés par unité.

    `derivees` est une liste de combinaisons `(canal A, opérateur, canal B)`.
    Elles rejoignent le même tracé que les canaux bruts, et se rangent dans le
    panneau de leur unité : deux sommes en N·m se comparent donc directement,
    sur la même échelle — ce qui est tout l'intérêt d'en tracer plusieurs.

    `zones` superpose les zones détectées pour CE fichier, restreintes aux
    familles de `types_zones` : confronter la détection à n'importe quel canal —
    les plages de repos au régime, par exemple — est le seul contrôle qui ne
    dépende pas du couple.
    """
    try:
        with st.spinner("Lecture du fichier (lecture seule)…"):
            signaux = lecteurs.charger_canaux(
                fichier, noms, float(st.session_state["freq"])
            )
    except Exception as exc:
        st.error(f"Chargement impossible : {exc}")
        return

    t = signaux.t
    if t.size < 2:
        st.warning("Fenêtre de temps commune trop courte pour un tracé.")
        return

    borne_min, borne_max = float(t[0]), float(t[-1])
    debut, fin = st.slider(
        "Plage de temps (s)", borne_min, borne_max, (borne_min, borne_max),
        key=f"vue::plage::{fichier.name}",
        help="Resserrer la plage est le seul moyen de voir un détail rapide : "
        "à l'échelle d'un cycle entier, quelques dizaines de millisecondes sont invisibles.",
    )
    fenetre = (t >= debut) & (t <= fin)
    if fenetre.sum() < 2:
        st.warning("Plage trop étroite.")
        return

    courbes = {nom: signaux.scalaires[nom][fenetre] for nom in noms
               if nom in signaux.scalaires}
    unites = {nom: catalogue.get(nom, "") for nom in courbes}

    for canal_a, operateur, canal_b in derivees:
        if canal_a not in courbes or canal_b not in courbes:
            continue
        signe = 1.0 if operateur == "+" else -1.0
        etiquette = f"{canal_a} {operateur} {canal_b}"
        courbes[etiquette] = courbes[canal_a] + signe * courbes[canal_b]
        unites[etiquette] = catalogue.get(canal_a, "")

    pas = max(1, int(np.ceil(fenetre.sum() / MAX_POINTS_TRACES)))
    figure = graphiques.figure_visualisation(
        t[fenetre][::pas],
        {nom: valeurs[::pas] for nom, valeurs in courbes.items()},
        unites,
        titre=f"{fichier.name}  —  {fin - debut:.1f} s affichées",
        decimation=pas,
        zones=zones,
        types_zones=types_zones,
    )
    st.pyplot(figure, use_container_width=True)

    tampon = io.BytesIO()
    figure.savefig(tampon, format="png", dpi=150, bbox_inches="tight")
    st.download_button(
        "Télécharger ce tracé (.png)", data=tampon.getvalue(),
        file_name=f"{fichier.stem}_visualisation.png", mime="image/png",
    )
    st.caption(signaux.diagnostic.resume())


def _appliquer_inventaire(inventaire: dict[str, dict]) -> None:
    """Pose les propositions de mapping et de type, sans écraser une saisie."""
    depart = E.etat_initial(inventaire)
    if not st.session_state["canaux_communs"]:
        st.session_state["canaux_communs"] = depart["canaux_communs"]
    for dossier, mapping in depart["canaux"].items():
        st.session_state["canaux"].setdefault(dossier, mapping)
    # Alimente les clés de widgets, sans toucher à celles déjà renseignées.
    for role, valeur in st.session_state["canaux_communs"].items():
        if role == "etat":
            st.session_state.setdefault("gcanal::etat", list(valeur or []))
        else:
            st.session_state.setdefault(f"gcanal::{role}", valeur or E.CANAL_ABSENT)
    for dossier, mapping in st.session_state["canaux"].items():
        for role, valeur in mapping.items():
            if role == "etat":
                st.session_state.setdefault(f"canal::{dossier}::etat", list(valeur or []))
            else:
                st.session_state.setdefault(f"canal::{dossier}::{role}", valeur or E.CANAL_ABSENT)


def _charger_configuration(chemin: Path) -> None:
    """Repeuple l'écran depuis un YAML enregistré."""
    brut = E.depuis_yaml(chemin)
    for cle, valeur in E.scalaires_depuis_dict(brut).items():
        if cle not in DEFAUTS:
            continue
        # `null` en YAML — un passe-haut désactivé, par exemple — ne peut pas
        # être posé tel quel dans un champ numérique : Streamlit lèverait. On
        # retombe sur la valeur qui, dans l'interface, désigne la désactivation.
        if valeur is None and isinstance(DEFAUTS[cle], (int, float)):
            valeur = type(DEFAUTS[cle])(0)
        st.session_state[cle] = valeur
    etat = E.etat_depuis_dict(brut)
    st.session_state["canaux"] = etat["canaux"]
    defaut = (brut.get("canaux") or {}).get("defaut") or {}
    if defaut:
        st.session_state["canaux_communs"] = defaut
        st.session_state["mapping_commun"] = True
        for role, valeur in defaut.items():
            if role == "etat":
                st.session_state["gcanal::etat"] = list(valeur or [])
            else:
                st.session_state[f"gcanal::{role}"] = valeur or E.CANAL_ABSENT
    else:
        st.session_state["mapping_commun"] = False
    for dossier, mapping in etat["canaux"].items():
        for role, valeur in mapping.items():
            if role == "etat":
                st.session_state[f"canal::{dossier}::etat"] = list(valeur or [])
            else:
                st.session_state[f"canal::{dossier}::{role}"] = valeur or E.CANAL_ABSENT


def _restaurer_memoire() -> None:
    """Repose la configuration de la dernière utilisation, une fois par session.

    Appelée AVANT le rendu des widgets : Streamlit refuse qu'on modifie la clé
    d'un widget déjà instancié.

    Un échec de lecture ne doit jamais empêcher l'outil de démarrer — la mémoire
    est un confort. Le motif est conservé pour être affiché, plutôt que la panne
    passée sous silence.
    """
    if st.session_state.get("memoire::lue"):
        return
    st.session_state["memoire::lue"] = True
    if not FICHIER_MEMOIRE.is_file():
        return
    try:
        _charger_configuration(FICHIER_MEMOIRE)
        st.session_state["memoire::restauree"] = True
    except Exception as exc:
        st.session_state["memoire::erreur"] = str(exc)


def _memoriser_configuration() -> None:
    """Retient la configuration courante pour le prochain lancement.

    Écrite en fin d'exécution, sans que l'utilisateur ait à y penser : le mapping
    n'a aucune raison de changer d'une campagne à l'autre, et le redemander à
    chaque lancement serait une corvée sans contrepartie.

    Deux garde-fous :
      * on n'écrit que si le contenu a changé, pour ne pas toucher le disque à
        chaque interaction — Streamlit réexécute le script en permanence ;
      * on n'écrase jamais une mémoire qui contient un mapping par une
        configuration qui n'en a pas. Sans cela, un simple lancement avant
        l'inventaire effacerait le travail de la veille.
    """
    configuration = _configuration_courante()
    if not configuration:
        return
    try:
        texte = E.vers_yaml(configuration)
        ancien = (
            FICHIER_MEMOIRE.read_text(encoding="utf-8")
            if FICHIER_MEMOIRE.is_file() else ""
        )
        if texte == ancien:
            return
        if ancien and not E.mapping_renseigne(configuration):
            if E.mapping_renseigne(E.depuis_yaml(FICHIER_MEMOIRE)):
                return
        FICHIER_MEMOIRE.parent.mkdir(parents=True, exist_ok=True)
        FICHIER_MEMOIRE.write_text(texte, encoding="utf-8")
    except Exception as exc:
        st.session_state["memoire::erreur"] = str(exc)


_restaurer_memoire()


# ---------------------------------------------------------------------------
# Barre latérale
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Corrélation couple\n**Transmissions ↔ banc GMP**")
    st.caption("Chapitre 10 — rapport AMDEC")
    st.divider()

    st.text_input(
        "Dossier racine des acquisitions", key="racine",
        placeholder=r"C:\user\SD17365\Documents",
        help="Le dossier qui contient les acquisitions. Les sous-dossiers, s'il y en "
        "a, sont parcourus aussi. Rien n'est envoyé hors du poste : tout s'exécute "
        "en local, en lecture seule.",
    )
    _choisir_dossier("racine", "des acquisitions")
    racine = st.session_state["racine"]
    racine_ok = bool(racine) and Path(racine).is_dir()
    if racine and not racine_ok:
        st.error("Dossier introuvable.")

    st.text_input("Dossier de sortie", key="dossier_sortie",
                  help="Rapport Markdown et figures PNG y seront écrits. "
                  "Il est créé s'il n'existe pas encore.")
    _choisir_dossier("dossier_sortie", "de sortie")

    st.divider()
    st.markdown("**Chaîne de mesure**")
    st.number_input("Pleine échelle capteur (N·m)", min_value=1.0, step=50.0, key="pe",
                    help="Toutes les grandeurs en « % PE » s'y rapportent.")
    st.selectbox(
        "Voie comparée à la référence", MODES_COMPARAISON, key="mode",
        help="Le mode s'applique de la même façon aux voies mesurées et aux voies "
        "de référence : « moyenne » compare (G+D)/2 mesuré à (G+D)/2 référence.",
    )
    st.number_input("Rapport de réduction appliqué à la référence", min_value=0.0001,
                    step=0.1, format="%.4f", key="rapport",
                    help="1,0 = référence prise au même point que les transmissions.")
    st.number_input("Fréquence de rééchantillonnage (Hz)", min_value=1.0, step=10.0,
                    key="freq",
                    help="Grille de temps commune. Au moins égale à la cadence du canal "
                    "le plus rapide utilisé.")
    _case(
        "Sorties du banc chargées symétriquement", "ligne_droite", False,
        "Condition pour que le résidu gauche − droite soit un indicateur métrologique "
        "et non un écart physique réel. À décocher si un essai sollicite le "
        "différentiel ou s'accompagne de patinage d'un seul côté.",
    )

    st.divider()
    st.markdown("**Configuration**")
    if st.session_state.get("memoire::restauree"):
        st.caption(
            "Reprise de la dernière utilisation : mapping des canaux, dossiers et "
            "hypothèses. Modifie ce que tu veux, tout est retenu à nouveau."
        )
    elif FICHIER_MEMOIRE.is_file():
        st.caption("Mémoire présente, sera reprise au prochain lancement.")
    else:
        st.caption("La configuration sera retenue pour le prochain lancement.")
    if st.session_state.get("memoire::erreur"):
        st.warning(f"Mémoire indisponible : {st.session_state['memoire::erreur']}")

    if FICHIER_MEMOIRE.is_file() and st.button(
        "Oublier la mémoire", use_container_width=True,
        help=f"Supprime {FICHIER_MEMOIRE}. Le prochain lancement repartira des "
        "valeurs par défaut.",
    ):
        FICHIER_MEMOIRE.unlink(missing_ok=True)
        st.session_state.pop("memoire::restauree", None)
        st.success("Mémoire effacée.")
        st.rerun()

    st.caption(f"Fichier de projet : `{FICHIER_CONFIG.relative_to(_RACINE_PROJET)}`")
    if FICHIER_CONFIG.is_file() and st.button("Recharger ce fichier",
                                              use_container_width=True):
        _charger_configuration(FICHIER_CONFIG)
        st.success("Configuration rechargée.")
        st.rerun()

    st.divider()
    st.markdown(
        "<div class='etape'>"
        + _pastille(racine_ok or None, "Dossier racine")
        + "<br>"
        + _pastille(bool(st.session_state["inventaire"]) or None, "Canaux inventoriés")
        + "<br>"
        + _pastille(st.session_state["campagne"] is not None or None, "Analyse exécutée")
        + "</div>",
        unsafe_allow_html=True,
    )


def _mapping_depuis_widgets(prefixe: str) -> dict[str, object]:
    mapping: dict[str, object] = {}
    for role in E.ORDRE_CANAUX:
        valeur = st.session_state.get(f"{prefixe}{role}", E.CANAL_ABSENT)
        mapping[role] = None if valeur == E.CANAL_ABSENT else valeur
    mapping["etat"] = list(st.session_state.get(f"{prefixe}etat", []))
    return mapping


def _configuration_courante() -> dict | None:
    inventaire = st.session_state["inventaire"]
    if not st.session_state["racine"]:
        return None
    thermique = dict(_section("t"))
    thermique["plage_service_C"] = (
        float(st.session_state["plage"]) if st.session_state["plage_connue"] else None
    )
    intercorrelation = dict(_section("i"))
    if not intercorrelation["passe_haut_Hz"]:
        intercorrelation["passe_haut_Hz"] = None
    # 0 s à l'écran signifie « règle automatique », que la configuration exprime
    # par `null` : une durée minimale nulle laisserait passer des fenêtres d'un
    # seul point.
    if not intercorrelation.get("duree_min_fenetre_s"):
        intercorrelation["duree_min_fenetre_s"] = None

    commun = st.session_state["mapping_commun"]
    return E.construire_dict(
        racine=st.session_state["racine"],
        dossier_sortie=st.session_state["dossier_sortie"],
        pleine_echelle_Nm=st.session_state["pe"],
        mode_comparaison=st.session_state["mode"],
        rapport_reduction=st.session_state["rapport"],
        incertitude_reference_k1_Nm=(
            float(st.session_state["u_ref"]) if st.session_state["u_ref_connue"] else None
        ),
        frequence_Hz=st.session_state["freq"],
        remontage_realise={"Oui": True, "Non": False, "Non précisé": None}[
            st.session_state["remontage"]
        ],
        canaux={} if commun else {
            d: _mapping_depuis_widgets(f"canal::{d}::") for d in inventaire
        },
        canaux_communs=_mapping_depuis_widgets("gcanal::") if commun else None,
        # Aucun essai déclaré : les zones sont découvertes automatiquement.
        essais={},
        ligne_droite=bool(st.session_state["ligne_droite"]),
        paliers=_section("p"),
        intercorrelation=intercorrelation,
        zero=_section("z"),
        thermique=thermique,
        diagnostic=_section("d"),
    )


onglets = st.tabs(
    ["1 · Exploration", "2 · Visualisation", "3 · Canaux", "4 · Zones détectées",
     "5 · Hypothèses", "6 · Analyse & résultats"]
)
EXPLORATION, VISUALISATION, CANAUX, ZONES, HYPOTHESES, ANALYSE = range(6)

# ---------------------------------------------------------------------------
# 1 · Exploration
# ---------------------------------------------------------------------------

with onglets[EXPLORATION]:
    st.subheader("Inventaire des canaux")
    st.markdown(
        "<div class='aide'>Lit un échantillon de fichiers et liste les canaux réellement "
        "présents, avec leurs unités et leurs cadences. C'est ce relevé qui sert à "
        "confirmer les libellés — rien n'est supposé.<br>"
        "Les acquisitions posées directement dans le dossier racine sont regroupées sous "
        "<code>(racine)</code> ; les sous-dossiers, s'il y en a, forment chacun un groupe "
        "— un simple classement d'affichage, sans effet sur les calculs.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    gauche, _ = st.columns([1, 3])
    with gauche:
        st.number_input("Fichiers examinés par dossier", min_value=1, max_value=20,
                        key="n_fichiers")
        lancer = st.button("Inventorier les canaux", type="primary",
                           disabled=not racine_ok, use_container_width=True)
        if st.button("Vider le cache", use_container_width=True, disabled=not racine_ok):
            inventorier.clear()
            st.success("Cache vidé.")

    if lancer:
        try:
            with st.spinner("Lecture des acquisitions (lecture seule)…"):
                st.session_state["inventaire"] = inventorier(
                    racine, int(st.session_state["n_fichiers"])
                )
            _appliquer_inventaire(st.session_state["inventaire"])
            st.rerun()  # rafraîchit l'indicateur d'étape, rendu avant ce point
        except Exception as exc:
            st.error(f"Inventaire impossible : {exc}")

    inventaire = st.session_state["inventaire"]
    if not inventaire:
        st.info("Renseigne le dossier racine, puis lance l'inventaire.")
    else:
        st.success(f"{len(inventaire)} sous-dossier(s) d'essai contenant des `.mf4`.")
        for dossier, donnees in inventaire.items():
            with st.expander(f"{dossier} — {donnees['n_fichiers']} fichier(s)"):
                if donnees["erreurs"]:
                    st.error("Erreurs de lecture :\n\n"
                             + "\n".join(f"- {e}" for e in donnees["erreurs"]))
                if not donnees["coherent"]:
                    st.warning(
                        "⚠️ Les fichiers examinés ne portent pas le même jeu de canaux. "
                        "Le mapping de ce dossier est à vérifier fichier par fichier."
                    )
                if donnees["n_groupes"] > 1:
                    st.info(
                        f"ℹ️ {donnees['n_groupes']} groupes de canaux, donc autant d'horloges "
                        "distinctes. Les canaux seront ramenés sur une grille de temps commune "
                        "(intersection des plages, aucune extrapolation)."
                    )
                st.caption("Fichiers examinés : "
                           + ", ".join(f"`{f}`" for f in donnees["fichiers"]))
                st.dataframe(donnees["canaux"], use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# 3 · Canaux
# ---------------------------------------------------------------------------

with onglets[VISUALISATION]:
    st.subheader("Visualisation des signaux")
    inventaire = st.session_state["inventaire"]
    if not inventaire:
        st.info("Lance d'abord l'inventaire (onglet 1).")
    else:
        st.markdown(
            "<div class='aide'>Trace n'importe quel canal en fonction du temps, fichier par "
            "fichier. Sert à vérifier qu'une acquisition contient bien ce qu'on croit, et que "
            "son allure correspond à ce que l'onglet « Zones détectées » y a repéré — un "
            "contrôle que l'analyse ne peut pas faire à votre place.<br>"
            "Les courbes sont <b>groupées par unité</b>, un panneau par unité : superposer un "
            "couple et un régime sur un même axe écraserait l'un des deux.</div>",
            unsafe_allow_html=True,
        )
        st.write("")

        c1, c2 = st.columns(2)
        with c1:
            dossier = st.selectbox("Essai", list(inventaire), key="vue::dossier")
        fichiers = fichiers_du_dossier(st.session_state["racine"], dossier)
        if not fichiers:
            st.warning("Aucune acquisition exploitable dans ce dossier.")
        else:
            with c2:
                choix = st.selectbox(
                    "Fichier", [f.name for f in fichiers], key="vue::fichier",
                    help=f"{len(fichiers)} acquisition(s) dans ce dossier.",
                )
            fichier = next(f for f in fichiers if f.name == choix)

            try:
                catalogue = canaux_du_fichier(str(fichier))
            except Exception as exc:
                catalogue = {}
                st.error(f"Lecture impossible : {exc}")

            if catalogue:
                noms = list(catalogue)
                # Proposition de départ : les canaux déjà mappés, s'ils
                # existent dans CE fichier.
                defaut = [
                    st.session_state.get(f"gcanal::{role}")
                    for role in E.ORDRE_CANAUX
                ]
                defaut = [n for n in defaut if isinstance(n, str) and n in noms][:3]
                selection = st.multiselect(
                    "Canaux à tracer", noms,
                    default=st.session_state.get("vue::canaux") or defaut,
                    key="vue::canaux",
                    help=f"{graphiques.MAX_COURBES} courbes au maximum : au-delà, deux séries "
                    "ne sont plus distinguables de façon fiable.",
                )

                st.markdown("**Courbes dérivées** — combinaisons de deux canaux")
                st.caption(
                    "Elles se tracent dans le **même graphe** que les canaux bruts, "
                    "et dans le panneau de leur unité : deux sommes en N·m se "
                    "comparent donc directement, sur la même échelle."
                )
                n_derivees = st.number_input(
                    "Nombre de combinaisons", min_value=0, max_value=MAX_DERIVEES,
                    step=1, key="vue::n_derivees",
                    help="0 pour n'en tracer aucune. Chaque combinaison ajoute une "
                    "courbe au tracé.",
                )

                derivees: list[tuple[str, str, str]] = []
                for i in range(int(n_derivees)):
                    d1, d2, d3 = st.columns([2, 0.7, 2])
                    with d1:
                        gauche = st.selectbox(
                            f"Canal A — combinaison {i + 1}", noms, key=f"vue::der::{i}::a",
                            index=min(i * 2, len(noms) - 1),
                        )
                    with d2:
                        operateur = st.selectbox("Opé.", ["+", "−"], key=f"vue::der::{i}::op")
                    with d3:
                        droite = st.selectbox(
                            f"Canal B — combinaison {i + 1}", noms, key=f"vue::der::{i}::b",
                            index=min(i * 2 + 1, len(noms) - 1),
                        )
                    if catalogue.get(gauche) != catalogue.get(droite):
                        st.warning(
                            f"« {gauche} » est en {catalogue.get(gauche) or 'sans unité'} et "
                            f"« {droite} » en {catalogue.get(droite) or 'sans unité'} : "
                            "leur combinaison n'a pas de sens physique."
                        )
                    derivees.append((gauche, operateur, droite))

                # Les canaux d'une combinaison sont chargés même s'ils ne sont pas
                # sélectionnés : sans eux, la courbe dérivée ne peut pas être calculée.
                a_tracer = list(dict.fromkeys(
                    list(selection) + [n for a, _, b in derivees for n in (a, b)]
                ))
                total = len(selection) + len(derivees)

                # --- zones détectées à superposer -------------------------
                zones_fichier = next(
                    (z for z in st.session_state["zones"]
                     if z.chemin.name == fichier.name and z.erreur is None),
                    None,
                )
                st.markdown("**Zones détectées à superposer**")
                if zones_fichier is None:
                    types_zones = []
                    st.caption(
                        "Lance « Détecter les zones » (onglet 4) pour pouvoir les "
                        "superposer ici."
                    )
                else:
                    # Seules les familles réellement présentes sont proposées :
                    # offrir « paliers — descente » sur un cycle qui n'en a pas
                    # ferait douter du réglage plutôt que du contenu.
                    presentes = zones_fichier.familles()
                    if not presentes:
                        types_zones = []
                        st.caption("Aucune zone détectée dans cette acquisition.")
                    else:
                        defaut = [c for c in graphiques.TYPES_ZONES_DEFAUT
                                  if c in presentes]
                        choix_zones = st.multiselect(
                            "Familles à afficher",
                            [graphiques.TYPES_ZONES[c] for c in presentes],
                            default=[graphiques.TYPES_ZONES[c] for c in defaut],
                            key=f"vue::zones::{fichier.name}",
                            help="Les paliers sont décochés par défaut : un balayage "
                            "en compte des dizaines, et les superposer tous rend le "
                            "tracé illisible.",
                        )
                        inverse = {v: k for k, v in graphiques.TYPES_ZONES.items()}
                        types_zones = [inverse[libelle] for libelle in choix_zones]
                        if not types_zones:
                            # Un sélecteur vide se lit comme une panne : on dit
                            # que le tracé est nu, et que c'est volontaire.
                            st.caption(
                                "Aucune zone superposée — choisis une famille "
                                "ci-dessus pour la voir sur le tracé."
                            )

                if not a_tracer:
                    st.info("Sélectionne au moins un canal.")
                elif total > graphiques.MAX_COURBES:
                    st.error(
                        f"{total} courbes demandées ({len(selection)} canaux et "
                        f"{len(derivees)} combinaison(s)), {graphiques.MAX_COURBES} au "
                        "maximum. Au-delà, deux courbes cessent d'être distinguables."
                    )
                else:
                    tracer_visualisation(
                        fichier, a_tracer, catalogue, derivees,
                        zones=zones_fichier if types_zones else None,
                        types_zones=types_zones,
                    )

with onglets[CANAUX]:
    st.subheader("Mapping des canaux")
    inventaire = st.session_state["inventaire"]
    if not inventaire:
        st.info("Lance d'abord l'inventaire (onglet 1).")
    else:
        _case(
            "Les libellés sont identiques sur toute la campagne (mapping commun)",
            "mapping_commun", True,
            "Cas courant. Décoche seulement si les noms de canaux diffèrent "
            "d'un dossier d'essai à l'autre.",
        )
        st.markdown(
            "<div class='aide'>Les valeurs pré-sélectionnées sont des <b>candidats</b> "
            "repérés par mots-clés, pas des certitudes : vérifie chaque ligne. "
            "« — absent — » rend explicitement non calculables les grandeurs qui "
            "dépendent de ce canal, plutôt que de les estimer.<br>"
            "La référence banc se déclare sur <b>deux voies gauche et droite</b> ; "
            "la voie unique en dessous n'est qu'un repli si le banc n'en fournit qu'une.</div>",
            unsafe_allow_html=True,
        )
        st.write("")

        if st.session_state["mapping_commun"]:
            options = [E.CANAL_ABSENT] + E.noms_tous_dossiers(inventaire)
            colonnes = st.columns(2)
            for i, role in enumerate(E.ORDRE_CANAUX):
                with colonnes[i % 2]:
                    _selectbox_canal(E.LIBELLES_CANAUX[role], f"gcanal::{role}", options)
            _multiselect_etat(
                "Canaux d'état (LED, CRC, erreurs) — relevés sans interprétation",
                "gcanal::etat", options[1:],
            )
            manquants = [
                r for r in ("couple_reference_gauche", "couple_reference_droite")
                if st.session_state.get(f"gcanal::{r}") == E.CANAL_ABSENT
            ]
            if len(manquants) == 2 and st.session_state.get("gcanal::couple_reference") == E.CANAL_ABSENT:
                st.warning("Aucune voie de couple de référence : aucun essai ne sera exploitable.")
            elif manquants and len(manquants) == 1:
                st.warning("Une seule voie de référence sur deux est renseignée.")
            # Les canaux communs doivent exister dans chaque dossier.
            for dossier, donnees in inventaire.items():
                absents = [
                    E.LIBELLES_CANAUX[r]
                    for r in E.ORDRE_CANAUX
                    if st.session_state.get(f"gcanal::{r}") not in (E.CANAL_ABSENT, None)
                    and st.session_state.get(f"gcanal::{r}") not in donnees["noms"]
                ]
                if absents:
                    st.warning(
                        f"« {dossier} » ne contient pas : {', '.join(absents)}. "
                        "Décoche le mapping commun pour traiter ce dossier à part."
                    )
        else:
            for dossier, donnees in inventaire.items():
                options = [E.CANAL_ABSENT] + donnees["noms"]
                with st.expander(dossier):
                    colonnes = st.columns(2)
                    for i, role in enumerate(E.ORDRE_CANAUX):
                        with colonnes[i % 2]:
                            _selectbox_canal(
                                E.LIBELLES_CANAUX[role], f"canal::{dossier}::{role}", options
                            )
                    _multiselect_etat("Canaux d'état", f"canal::{dossier}::etat", options[1:])

# ---------------------------------------------------------------------------
# 4 · Zones détectées
# ---------------------------------------------------------------------------

with onglets[ZONES]:
    st.subheader("Zones détectées")
    inventaire = st.session_state["inventaire"]
    if not st.session_state["racine"]:
        st.info("Renseigne le dossier racine (barre latérale).")
    else:
        st.markdown(
            "<div class='aide'>Aucun type d'essai n'est à déclarer : chaque acquisition est "
            "examinée pour <b>ce qu'elle contient réellement</b> — paliers stabilisés, plage "
            "dynamique, relevés de zéro — et chaque grandeur est calculée à partir des zones "
            "qui la concernent, tous fichiers confondus.<br>"
            "Le tableau dit <b>combien</b>, les figures disent <b>où</b>. "
            "<b>Automatique ne veut pas dire opaque</b> : si une acquisition n'alimente pas "
            "ce que vous attendiez, les seuils de détection se règlent dans l'onglet "
            "Hypothèses.</div>",
            unsafe_allow_html=True,
        )
        st.write("")

        if st.button("Détecter les zones", type="primary", use_container_width=False):
            configuration = _configuration_courante()
            cfg, erreurs = E.valider(configuration) if configuration else (None, ["configuration incomplète"])
            if erreurs or cfg is None:
                st.error("Configuration invalide :\n\n" + "\n".join(f"- {e}" for e in erreurs))
            else:
                try:
                    with st.spinner("Lecture des acquisitions (lecture seule)…"):
                        trouvees, signaux_detectes = detecter_zones(cfg)
                    st.session_state["zones"] = trouvees
                    st.session_state["signaux_zones"] = signaux_detectes
                    st.rerun()
                except Exception as exc:
                    st.error(f"Détection impossible : {exc}")

        zones = st.session_state["zones"]
        if not zones:
            st.info("Lance la détection pour voir ce que contiennent tes acquisitions.")
        else:
            synthese = Zn.synthese(zones)
            colonnes = st.columns(5)
            for colonne, (libelle, cle) in zip(colonnes, (
                ("Acquisitions", "fichiers"), ("Régression", "régression"),
                ("Hystérésis", "hystérésis"), ("Retard", "retard"),
                ("Dérive de zéro", "dérive de zéro"),
            )):
                colonne.metric(libelle, synthese[cle])

            st.dataframe(
                [
                    {
                        "Acquisition": z.chemin.name,
                        # Placée juste après le nom : c'est la conclusion du relevé,
                        # les colonnes suivantes n'en sont que la justification.
                        "Alimente": ", ".join(z.contributions()) or "—",
                        "Durée (s)": round(z.duree_s, 1),
                        "Paliers": len(z.paliers),
                        "Niveaux": z.niveaux_distincts,
                        "Montée/desc.": "oui" if z.a_montee_et_descente else "—",
                        # Texte plutôt que nombre : « — » se lit comme une absence,
                        # là où une case vide laisserait croire à un relevé manquant.
                        "Dynamique (s)": (
                            f"{z.duree_dynamique_s:.0f}" if z.alimente_retard else "—"
                        ),
                        "Zéros déb./fin": "oui" if z.alimente_derive_zero else "—",
                    }
                    for z in zones
                ],
                use_container_width=True, hide_index=True,
            )

            sans_contribution = [z for z in zones if not z.contributions() and z.erreur is None]
            if sans_contribution:
                st.warning(
                    f"{len(sans_contribution)} acquisition(s) n'alimentent aucune grandeur : "
                    + ", ".join(f"`{z.chemin.name}`" for z in sans_contribution[:5])
                    + ". Vérifie le mapping, ou desserre les seuils de détection."
                )
            for z in zones:
                if z.erreur:
                    st.error(f"`{z.chemin.name}` : {z.erreur}")

            signaux = st.session_state.get("signaux_zones") or {}
            if signaux:
                st.divider()
                with st.expander("Où se trouvent ces zones", expanded=True):
                    st.markdown(
                        "<div class='aide'>Le tracé situe chaque zone sur le signal, en aplat "
                        "de fond et par type. C'est le contrôle qui manque au tableau : un "
                        "palier posé sur un transitoire ou une plage dynamique qui déborde sur "
                        "un arrêt ne se voient que là.<br>"
                        "Les bandes de palier montrent la part <b>effectivement moyennée</b> — "
                        "la fraction finale de la plage stable — et non toute la plage.</div>",
                        unsafe_allow_html=True,
                    )
                    st.write("")
                    lisibles = [z for z in zones if z.erreur is None]
                    g, d = st.columns([1, 1])
                    with g:
                        choix_figure = st.selectbox(
                            "Acquisition", [z.chemin.name for z in lisibles],
                            key="zones::apercu",
                            help="Les figures sont aussi écrites en PNG dans le dossier "
                            "de sortie au moment de l'analyse — toutes familles affichées.",
                        )
                    zone_choisie = next(
                        (z for z in lisibles if z.chemin.name == choix_figure), None
                    )
                    presentes = zone_choisie.familles() if zone_choisie else []
                    with d:
                        libelles = st.multiselect(
                            "Familles à afficher",
                            [graphiques.TYPES_ZONES[c] for c in presentes],
                            default=[graphiques.TYPES_ZONES[c] for c in presentes],
                            key="zones::familles",
                            help="Tout est affiché par défaut : c'est l'objet de cet "
                            "onglet. Décoche les paliers si leur nombre masque le reste.",
                        )
                    inverse = {v: k for k, v in graphiques.TYPES_ZONES.items()}
                    familles = [inverse[x] for x in libelles]

                    if zone_choisie is not None:
                        image = figure_zones_png(
                            choix_figure, zone_choisie, signaux,
                            float(st.session_state["pe"]), familles,
                        )
                        if image:
                            st.image(image, use_container_width=True)
                            st.download_button(
                                "Télécharger cette figure (.png)", data=image,
                                file_name=f"zones_{Path(choix_figure).stem}.png",
                                mime="image/png",
                            )
                    if _case("Afficher toutes les acquisitions", "zones::toutes", False,
                             "Utile pour balayer une campagne entière d'un coup d'œil. "
                             "Chaque acquisition garde les familles qu'elle contient."):
                        for z in lisibles:
                            if z.chemin.name == choix_figure:
                                continue
                            autre = figure_zones_png(
                                z.chemin.name, z, signaux,
                                float(st.session_state["pe"]),
                                [c for c in familles if c in z.familles()] or None,
                            )
                            if autre:
                                st.image(autre, use_container_width=True)

# ---------------------------------------------------------------------------
# 5 · Hypothèses
# ---------------------------------------------------------------------------

with onglets[HYPOTHESES]:
    st.subheader("Hypothèses de traitement et saisies utilisateur")
    st.markdown(
        "<div class='aide'>Ces valeurs sont reprises telles quelles dans le § Hypothèses "
        "du rapport. Les deux premières ne peuvent pas être déduites des acquisitions : "
        "laissées vides, leur contribution est exclue et l'incertitude élargie est "
        "annoncée comme un minorant.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.checkbox("Incertitude du banc connue", key="u_ref_connue",
                    help="À reprendre du certificat d'étalonnage du banc GMP.")
        st.number_input("Incertitude-type k=1 (N·m)", min_value=0.0, step=0.5, key="u_ref",
                        disabled=not st.session_state["u_ref_connue"],
                        help="Incertitude-TYPE (k=1). Si le certificat donne une incertitude "
                        "élargie à k=2, saisir la moitié.")
    with c2:
        st.checkbox("Plage de service connue", key="plage_connue",
                    help="Domaine d'emploi. L'excursion constatée sur les essais ne peut pas "
                    "en tenir lieu : elle décrit la campagne, pas les conditions de service.")
        st.number_input("Plage de température de service (°C)", min_value=0.0, step=5.0,
                        key="plage", disabled=not st.session_state["plage_connue"])
    with c3:
        st.selectbox("Démontage/remontage sur la campagne",
                     ["Non précisé", "Oui", "Non"], key="remontage",
                     help="« Non » motive la non-calculabilité par l'absence d'objet à "
                     "mesurer, plutôt que par une configuration incomplète.")

    st.divider()
    a, b = st.columns(2)
    with a:
        st.markdown("**Détection des paliers stabilisés**")
        st.number_input("Durée minimale d'un palier (s)", 0.2, 60.0, step=0.5,
                        key="p::duree_palier_s")
        st.number_input("Tolérance de stabilité (% PE)", 0.01, 10.0, step=0.1,
                        key="p::tolerance_stab_pc_pe")
        st.slider("Fraction finale moyennée", 0.1, 1.0, step=0.05, key="p::fraction_finale")
        st.number_input("Écart minimal entre paliers (% PE)", 0.1, 20.0, step=0.1,
                        key="p::ecart_min_paliers_pc_pe")
        st.number_input("Tolérance d'appariement (% PE)", 0.1, 20.0, step=0.1,
                        key="p::tolerance_appariement_pc_pe")
        st.markdown("**Relevés de zéro**")
        st.number_input("Durée de fenêtre de repos (s)", 0.5, 120.0, step=1.0,
                        key="z::duree_fenetre_s")
        st.number_input("Seuil de couple au repos (% PE)", 0.01, 10.0, step=0.1,
                        key="z::seuil_couple_ref_pc_pe")
        st.number_input("Seuil de régime au repos", 0.0, 5000.0, step=5.0,
                        key="z::seuil_regime")
        st.slider("Position exigée des relevés (fraction de l'essai)", 0.05, 0.5, step=0.05,
                  key="z::fraction_bord")
    with b:
        st.markdown("**Recalage temporel**")
        st.number_input("Retard maximal recherché (ms)", 1.0, 5000.0, step=50.0,
                        key="i::retard_max_ms")
        st.number_input("Passe-haut avant corrélation (Hz) — 0 pour désactiver",
                        0.0, 50.0, step=0.1, key="i::passe_haut_Hz")
        st.number_input("Seuil d'activité dynamique (% PE)", 0.1, 50.0, step=0.5,
                        key="i::seuil_activite_pc_pe")
        st.number_input("Fenêtre d'activité (s)", 0.1, 30.0, step=0.5,
                        key="i::fenetre_activite_s")
        st.number_input("Comblement des interruptions (s)", 0.0, 60.0, step=1.0,
                        key="i::duree_comblement_s")
        st.number_input(
            "Durée minimale d'une fenêtre (s) — 0 = automatique", 0.0, 120.0, step=1.0,
            key="i::duree_min_fenetre_s",
            help="0 applique la règle par défaut : dix fois le retard maximal "
            "recherché, avec un plancher de 2 s. Sur une campagne de départs "
            "arrêtés, ce verrou écarte souvent les fronts eux-mêmes, actifs "
            "seulement deux à quatre secondes — c'est le premier réglage à "
            "abaisser si les fronts n'apparaissent pas comme fenêtres.",
        )
        st.number_input(
            "Score minimal d'une fenêtre (fraction de la meilleure)", 0.0, 1.0,
            step=0.05, key="i::fraction_score_min",
            help="Toutes les fenêtres dynamiques sont corrélées, et le retard "
            "est la médiane des leurs. Celles dont le score amplitude × √durée "
            "tombe sous cette fraction de la meilleure sont écartées : sur un "
            "relevé long, une portion de bruit franchirait le seuil d'activité "
            "sans rien apprendre du retard.",
        )
        st.number_input(
            "Sous-fenêtres de contrôle du retard", 2, 10, step=1,
            key="i::n_blocs_coherence",
            help="Le retard est ré-estimé sur autant de sous-fenêtres égales. "
            "Moins, la mesure est trop bruitée pour trancher ; plus, chaque "
            "sous-fenêtre devient trop courte et des essais exploitables seraient "
            "signalés à tort.",
        )
        st.number_input(
            "Accord exigé entre sous-fenêtres (ms)", 1.0, 500.0, step=5.0,
            key="i::accord_blocs_max_ms",
            help="Au-delà de cette étendue, le retard est annoncé « faiblement "
            "identifié » : il ne tient qu'à une partie du signal. Cas typique du "
            "départ arrêté, où seul le front porte la synchronisation.",
        )
        st.markdown("**Sensibilité thermique**")
        st.number_input("Excursion thermique minimale (°C)", 0.5, 100.0, step=1.0,
                        key="t::amplitude_min_C")
        st.markdown("**Seuils de rédaction de la conclusion**")
        st.number_input("Offset significatif (% PE)", 0.01, 10.0, step=0.1,
                        key="d::seuil_offset_pc_pe")
        st.number_input("Gain significatif (%)", 0.01, 20.0, step=0.1, key="d::seuil_gain_pc")
        st.number_input("Retard significatif (ms)", 0.1, 500.0, step=1.0,
                        key="d::seuil_retard_ms")
        st.slider("Baisse de RMS attribuée à la synchronisation", 0.05, 0.95, step=0.05,
                  key="d::gain_rms_recalage")
        st.slider("|r| résidu / point de fonctionnement", 0.1, 0.95, step=0.05,
                  key="d::seuil_correlation_point_fct")

# ---------------------------------------------------------------------------
# 6 · Analyse & résultats
# ---------------------------------------------------------------------------


with onglets[ANALYSE]:
    st.subheader("Analyse")
    configuration = _configuration_courante()

    if configuration is None:
        st.info("Complète les onglets 1 à 5.")
    else:
        cfg, erreurs = E.valider(configuration)
        if erreurs:
            st.error("Configuration invalide :\n\n" + "\n".join(f"- {e}" for e in erreurs))
        else:
            for avertissement in cfg.verifier_mapping():
                st.warning(avertissement)
            for avertissement in cfg.verifier_saisies():
                st.info(avertissement)

            c1, c2, c3 = st.columns(3)
            with c1:
                lancer_analyse = st.button("Lancer l'analyse", type="primary",
                                           use_container_width=True)
            with c2:
                if st.button("Enregistrer la configuration", use_container_width=True,
                             help="Écrit config/correlation.yaml sur le poste. "
                             "Un rafraîchissement de page ne coûtera plus la saisie."):
                    FICHIER_CONFIG.parent.mkdir(parents=True, exist_ok=True)
                    FICHIER_CONFIG.write_text(E.vers_yaml(configuration), encoding="utf-8")
                    st.success(f"Enregistré : {FICHIER_CONFIG}")
            with c3:
                st.download_button(
                    "Télécharger (.yaml)",
                    data=E.vers_yaml(configuration).encode("utf-8"),
                    file_name="correlation.yaml", mime="text/yaml",
                    use_container_width=True,
                    help="Rejoue à l'identique : "
                    "python scripts/02_analyse.py --config correlation.yaml",
                )

            if lancer_analyse:
                try:
                    with st.status("Analyse en cours…", expanded=True) as suivi:
                        st.write("Lecture des acquisitions (lecture seule) et traitements…")
                        campagne = A.analyser(cfg)
                        st.write("Rédaction du rapport et des figures…")
                        chemin = R.ecrire(campagne)
                        suivi.update(label="Analyse terminée", state="complete", expanded=False)
                    st.session_state["campagne"] = campagne
                    st.session_state["rapport_chemin"] = str(chemin)
                    st.rerun()
                except Exception as exc:
                    st.session_state["campagne"] = None
                    st.error(f"Échec de l'analyse : {exc}")

    campagne = st.session_state["campagne"]
    if campagne is not None:
        pe = campagne.config.pleine_echelle_Nm
        st.divider()

        # Les tuiles lisent les grandeurs consolidées de la campagne : elles sont
        # renseignées de la même façon que l'essai soit déclaré ou que la zone
        # ait été découverte, donc l'affichage ne dépend pas du mode.
        reg = campagne.regression_globale
        if reg is not None and reg.non_calculable:
            reg = None
        retards = campagne.retards_ms
        inc = campagne.incertitude

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Offset b",
                  _valeur(100 * reg.b / pe, 3, True, " % PE") if reg else "non calculable")
        m2.metric("Erreur de sensibilité",
                  _valeur(100 * (reg.a - 1), 3, True, " %") if reg else "non calculable")
        # Même médiane que le rapport : la tuile et le tableau ne doivent pas
        # afficher deux chiffres différents pour la même grandeur.
        m3.metric("Retard temporel",
                  _valeur(float(np.median(retards)), 1, True, " ms")
                  if retards else "non calculable")
        m4.metric("Incertitude élargie (k=2)",
                  _valeur(inc.U_k2_pc_pe, 3, False, " % PE")
                  if inc and not inc.non_calculable else "non calculable")
        if inc and inc.minorant:
            # Pas de `delta=` : Streamlit y accolerait une flèche de tendance, qui
            # se lirait comme une hausse alors qu'il s'agit d'un avertissement.
            m4.markdown(
                f"<span style='color:{ATTENTION}'>▲</span> "
                f"<span style='color:{ENCRE_2};font-size:0.84rem'>minorant — "
                f"{len(inc.exclusions)} contribution(s) exclue(s)</span>",
                unsafe_allow_html=True,
            )

        st.divider()
        detail = ("Détail par acquisition" if campagne.mode == "automatique"
                  else "Détail par essai")
        vues = st.tabs(["Tableau récapitulatif", "Conclusions", "Figures",
                        "Incertitude", "Cartes de contrôle", detail, "Export"])
        with vues[0]:
            st.markdown(R.tableau_recapitulatif(campagne))
        with vues[1]:
            st.markdown(R.section_conclusions(campagne))
        with vues[2]:
            figures = campagne.figures + [f for e in campagne.essais for f in e.figures]
            if not figures:
                st.info("Aucune figure produite.")
            for i in range(0, len(figures), 2):
                colonnes = st.columns(2)
                for colonne, figure in zip(colonnes, figures[i : i + 2]):
                    with colonne:
                        st.image(str(figure), use_container_width=True)
                        st.caption(f"`{figure.name}`")
        with vues[3]:
            st.markdown(R.section_incertitude(campagne))
        with vues[4]:
            st.markdown(R.section_spc(campagne))
        with vues[5]:
            st.markdown(R.section_detail_essais(campagne))
        with vues[6]:
            st.download_button(
                "Rapport Markdown (.md)", data=R.rediger(campagne).encode("utf-8"),
                file_name="chapitre10_correlation.md", mime="text/markdown",
                use_container_width=True,
            )
            figures = campagne.figures + [f for e in campagne.essais for f in e.figures]
            if figures:
                tampon = io.BytesIO()
                with zipfile.ZipFile(tampon, "w", zipfile.ZIP_DEFLATED) as archive:
                    for figure in figures:
                        archive.write(figure, arcname=figure.name)
                st.download_button(
                    f"Figures PNG ({len(figures)}) — archive .zip", data=tampon.getvalue(),
                    file_name="figures_chapitre10.zip", mime="application/zip",
                    use_container_width=True,
                )
            st.caption(f"Écrit sur le poste : `{st.session_state.get('rapport_chemin', '')}`")


# ---------------------------------------------------------------------------
# Mémoire de la session
# ---------------------------------------------------------------------------

# En dernier, quand tous les widgets ont été rendus et que `session_state`
# reflète l'écran tel qu'il est : la configuration retenue est exactement celle
# que l'utilisateur voit.
_memoriser_configuration()
