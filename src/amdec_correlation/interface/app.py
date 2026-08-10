"""Interface graphique Streamlit du pipeline de corrélation couple.

Cette page n'implémente aucun traitement : elle assemble une configuration,
appelle la bibliothèque et affiche ce qui en sort. Les chiffres à l'écran sont
donc les mêmes que ceux du rapport, et la configuration exportée rejoue à
l'identique en ligne de commande.

Lancement :  python scripts/03_interface.py
"""

from __future__ import annotations

import io
import math
import sys
import zipfile
from pathlib import Path

import streamlit as st

_RACINE_PAQUET = Path(__file__).resolve().parents[3]
if str(_RACINE_PAQUET / "src") not in sys.path:
    sys.path.insert(0, str(_RACINE_PAQUET / "src"))

from amdec_correlation import analyse as A  # noqa: E402
from amdec_correlation import inventaire as I  # noqa: E402
from amdec_correlation import rapport as R  # noqa: E402
from amdec_correlation.config import CANAUX_SCALAIRES, MODES_COMPARAISON  # noqa: E402
from amdec_correlation.interface import etat as E  # noqa: E402

# Palette partagée avec les figures : l'écran et les PNG se lisent comme un seul
# système visuel.
ENCRE_2 = "#52514e"
ATTENUE = "#898781"
BON = "#0ca30c"
ATTENTION = "#fab219"
CRITIQUE = "#d03b3b"


# ---------------------------------------------------------------------------
# Mise en page
# ---------------------------------------------------------------------------

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
      /* Filets discrets, une nuance au-dessus du fond : jamais de pointillés. */
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


def _etat(cle: str, defaut):
    if cle not in st.session_state:
        st.session_state[cle] = defaut
    return st.session_state[cle]


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
            "candidats": I.candidats(inv.noms_uniques),
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
            "n_groupes": len(
                {d.groupe for desc in inv.canaux.values() for d in desc}
            ),
        }
    return resultats


def _valeur(x: float, decimales: int = 3, signe: bool = False, suffixe: str = "") -> str:
    if x is None or not math.isfinite(x):
        return "non calculable"
    gabarit = f"{{:+.{decimales}f}}" if signe else f"{{:.{decimales}f}}"
    return gabarit.format(x) + suffixe


# ---------------------------------------------------------------------------
# Barre latérale
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Corrélation couple\n**Transmissions ↔ banc GMP**")
    st.caption("Chapitre 10 — rapport AMDEC")
    st.divider()

    racine = st.text_input(
        "Dossier racine des acquisitions",
        value=_etat("racine", ""),
        placeholder=r"C:\Users\SD17365\Documents",
        help="Le dossier qui contient les sous-dossiers d'essai. Rien n'est envoyé "
        "hors du poste : tout s'exécute en local, en lecture seule.",
    )
    st.session_state["racine"] = racine
    racine_ok = bool(racine) and Path(racine).is_dir()
    if racine and not racine_ok:
        st.error("Dossier introuvable.")

    dossier_sortie = st.text_input(
        "Dossier de sortie",
        value=_etat("dossier_sortie", "sortie"),
        help="Rapport Markdown et figures PNG y seront écrits.",
    )
    st.session_state["dossier_sortie"] = dossier_sortie

    st.divider()
    st.markdown("**Chaîne de mesure**")
    pleine_echelle = st.number_input(
        "Pleine échelle capteur (N·m)", min_value=1.0, value=_etat("pe", 1500.0), step=50.0,
        help="Toutes les grandeurs en « % PE » s'y rapportent.",
    )
    st.session_state["pe"] = pleine_echelle

    mode = st.selectbox(
        "Voie comparée à la référence",
        MODES_COMPARAISON,
        index=MODES_COMPARAISON.index(_etat("mode", "moyenne")),
        help="« moyenne » convient à une référence prise au même point de la chaîne "
        "(rapport 1:1) ; « somme » si la référence est le couple total aux roues.",
    )
    st.session_state["mode"] = mode

    rapport_reduction = st.number_input(
        "Rapport de réduction appliqué à la référence",
        min_value=0.0001, value=_etat("rapport", 1.0), step=0.1, format="%.4f",
        help="1,0 = référence prise au même point que les transmissions.",
    )
    st.session_state["rapport"] = rapport_reduction

    frequence = st.number_input(
        "Fréquence de rééchantillonnage (Hz)",
        min_value=1.0, value=_etat("freq", 100.0), step=10.0,
        help="Grille de temps commune. À choisir au moins égale à la cadence du canal "
        "le plus rapide utilisé (visible dans l'onglet Exploration).",
    )
    st.session_state["freq"] = frequence

    st.divider()
    inventaire_fait = bool(st.session_state.get("inventaire"))
    campagne_faite = st.session_state.get("campagne") is not None
    st.markdown(
        "<div class='etape'>"
        + _pastille(racine_ok or None, "Dossier racine")
        + "<br>"
        + _pastille(inventaire_fait or None, "Canaux inventoriés")
        + "<br>"
        + _pastille(campagne_faite or None, "Analyse exécutée")
        + "</div>",
        unsafe_allow_html=True,
    )


onglets = st.tabs(
    [
        "1 · Exploration",
        "2 · Canaux",
        "3 · Essais",
        "4 · Hypothèses",
        "5 · Analyse & résultats",
    ]
)

# ---------------------------------------------------------------------------
# 1 · Exploration
# ---------------------------------------------------------------------------

with onglets[0]:
    st.subheader("Inventaire des canaux")
    st.markdown(
        "<div class='aide'>Lit un échantillon de fichiers par sous-dossier et liste "
        "les canaux réellement présents, avec leurs unités et leurs cadences. "
        "C'est ce relevé qui sert à confirmer les libellés — rien n'est supposé.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    gauche, droite = st.columns([1, 3])
    with gauche:
        n_fichiers = st.number_input(
            "Fichiers examinés par dossier", min_value=1, max_value=20,
            value=_etat("n_fichiers", 2),
        )
        st.session_state["n_fichiers"] = n_fichiers
        lancer = st.button("Inventorier les canaux", type="primary", disabled=not racine_ok,
                           use_container_width=True)
        if st.button("Vider le cache", use_container_width=True, disabled=not racine_ok):
            inventorier.clear()
            st.success("Cache vidé.")

    if lancer:
        try:
            with st.spinner("Lecture des acquisitions (lecture seule)…"):
                st.session_state["inventaire"] = inventorier(racine, int(n_fichiers))
            depart = E.etat_initial(st.session_state["inventaire"])
            st.session_state.setdefault("canaux", depart["canaux"])
            st.session_state.setdefault("essais", depart["essais"])
            # La barre latérale est rendue avant le contenu : sans ce rejeu, son
            # indicateur d'étape afficherait l'état d'avant l'inventaire.
            st.rerun()
        except Exception as exc:
            st.error(f"Inventaire impossible : {exc}")

    inventaire = st.session_state.get("inventaire") or {}
    if not inventaire:
        st.info("Renseigne le dossier racine, puis lance l'inventaire.")
    else:
        st.success(f"{len(inventaire)} sous-dossier(s) d'essai contenant des `.mf4`.")
        for dossier, donnees in inventaire.items():
            titre = f"{dossier} — {donnees['n_fichiers']} fichier(s)"
            with st.expander(titre, expanded=False):
                if donnees["erreurs"]:
                    st.error("Erreurs de lecture :\n\n" + "\n".join(f"- {e}" for e in donnees["erreurs"]))
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
                st.caption("Fichiers examinés : " + ", ".join(f"`{f}`" for f in donnees["fichiers"]))
                st.dataframe(donnees["canaux"], use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# 2 · Canaux
# ---------------------------------------------------------------------------

with onglets[1]:
    st.subheader("Mapping des canaux")
    inventaire = st.session_state.get("inventaire") or {}
    if not inventaire:
        st.info("Lance d'abord l'inventaire (onglet 1).")
    else:
        st.markdown(
            "<div class='aide'>Les valeurs pré-sélectionnées sont des <b>candidats</b> "
            "repérés par mots-clés, pas des certitudes : vérifie chaque ligne. "
            "« — absent — » rend explicitement non calculables les grandeurs qui "
            "dépendent de ce canal, plutôt que de les estimer.</div>",
            unsafe_allow_html=True,
        )
        st.write("")

        canaux_etat = _etat("canaux", E.etat_initial(inventaire)["canaux"])
        dossiers = list(inventaire)

        modele = st.selectbox(
            "Recopier le mapping d'un dossier vers tous les autres",
            ["—"] + dossiers,
            help="Cas courant : la même convention de nommage sur toute la campagne.",
        )
        if modele != "—" and st.button("Appliquer à tous les dossiers"):
            source = dict(canaux_etat.get(modele, {}))
            for dossier in dossiers:
                disponibles = set(inventaire[dossier]["noms"])
                # On ne recopie que ce qui existe réellement dans le dossier cible.
                copie = {
                    role: (source.get(role) if source.get(role) in disponibles else None)
                    for role in CANAUX_SCALAIRES
                }
                copie["etat"] = [c for c in (source.get("etat") or []) if c in disponibles]
                canaux_etat[dossier] = copie
            st.session_state["canaux"] = canaux_etat
            st.success(f"Mapping de « {modele} » recopié, en ne gardant que les canaux présents.")
            st.rerun()

        for dossier in dossiers:
            noms = inventaire[dossier]["noms"]
            options = [E.CANAL_ABSENT] + noms
            mapping = canaux_etat.setdefault(dossier, E.mapping_propose(noms))
            manquants = [r for r in ("couple_reference",) if not mapping.get(r)]
            marque = " ⚠️" if manquants else ""
            with st.expander(f"{dossier}{marque}", expanded=bool(manquants)):
                colonnes = st.columns(2)
                for i, role in enumerate(CANAUX_SCALAIRES):
                    courant = mapping.get(role)
                    index = options.index(courant) if courant in options else 0
                    with colonnes[i % 2]:
                        choix = st.selectbox(
                            E.LIBELLES_CANAUX[role], options, index=index,
                            key=f"canal::{dossier}::{role}",
                        )
                    mapping[role] = None if choix == E.CANAL_ABSENT else choix
                mapping["etat"] = st.multiselect(
                    "Canaux d'état (LED, CRC, erreurs) — relevés sans interprétation",
                    noms,
                    default=[c for c in (mapping.get("etat") or []) if c in noms],
                    key=f"etat::{dossier}",
                )
                if not mapping.get("couple_reference"):
                    st.warning("Sans couple de référence, cet essai ne sera pas exploité.")
                elif not (mapping.get("couple_mesure_gauche") or mapping.get("couple_mesure_droite")):
                    st.warning("Sans voie de couple mesuré, cet essai ne sera pas exploité.")
        st.session_state["canaux"] = canaux_etat

# ---------------------------------------------------------------------------
# 3 · Essais
# ---------------------------------------------------------------------------

with onglets[2]:
    st.subheader("Déclaration des essais")
    inventaire = st.session_state.get("inventaire") or {}
    if not inventaire:
        st.info("Lance d'abord l'inventaire (onglet 1).")
    else:
        st.markdown(
            "<div class='aide'>Le <b>type</b> détermine le traitement appliqué. "
            "<b>Ligne droite</b> : cocher si les deux transmissions voient le même couple "
            "par construction (sur banc, sorties chargées symétriquement) — c'est la condition "
            "pour que le résidu gauche − droite soit un indicateur métrologique et non un "
            "écart physique réel. <b>Groupe de remontage</b> : deux étiquettes distinctes sont "
            "nécessaires pour que la répétabilité après remontage existe.</div>",
            unsafe_allow_html=True,
        )
        st.write("")

        essais_etat = _etat("essais", E.etat_initial(inventaire)["essais"])
        types = list(E.LIBELLES_TYPES)

        for dossier in inventaire:
            declaration = essais_etat.setdefault(
                dossier, {"type": E.type_propose(dossier), "ligne_droite": False,
                          "groupe_remontage": None}
            )
            c1, c2, c3 = st.columns([3, 1, 1.4])
            with c1:
                declaration["type"] = st.selectbox(
                    dossier, types,
                    index=types.index(declaration.get("type", "dynamique")),
                    format_func=lambda t: E.LIBELLES_TYPES[t],
                    key=f"type::{dossier}",
                )
            with c2:
                declaration["ligne_droite"] = st.checkbox(
                    "Ligne droite", value=bool(declaration.get("ligne_droite")),
                    key=f"ld::{dossier}",
                )
            with c3:
                saisi = st.text_input(
                    "Groupe remontage", value=declaration.get("groupe_remontage") or "",
                    placeholder="avant / apres", key=f"gr::{dossier}",
                )
                declaration["groupe_remontage"] = saisi.strip() or None
        st.session_state["essais"] = essais_etat

        etiquettes = {d["groupe_remontage"] for d in essais_etat.values() if d.get("groupe_remontage")}
        if len(etiquettes) < 2:
            st.caption(
                f"ℹ️ {len(etiquettes)} groupe(s) de remontage déclaré(s). La répétabilité après "
                "remontage sera annoncée non calculable — motivée comme grandeur non définie "
                "si tu déclares ci-dessous qu'aucun remontage n'a eu lieu."
            )

# ---------------------------------------------------------------------------
# 4 · Hypothèses
# ---------------------------------------------------------------------------

with onglets[3]:
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
        u_ref_saisie = st.checkbox(
            "Incertitude du banc connue", value=_etat("u_ref_connue", False),
            help="À reprendre du certificat d'étalonnage du banc GMP.",
        )
        st.session_state["u_ref_connue"] = u_ref_saisie
        u_ref = st.number_input(
            "Incertitude-type k=1 (N·m)", min_value=0.0, value=_etat("u_ref", 2.0), step=0.5,
            disabled=not u_ref_saisie,
            help="Incertitude-TYPE (k=1). Si le certificat donne une incertitude élargie "
            "à k=2, saisir la moitié.",
        )
        st.session_state["u_ref"] = u_ref
    with c2:
        plage_saisie = st.checkbox(
            "Plage de service connue", value=_etat("plage_connue", False),
            help="Domaine d'emploi. L'excursion constatée sur les essais ne peut pas en "
            "tenir lieu : elle décrit la campagne, pas les conditions de service.",
        )
        st.session_state["plage_connue"] = plage_saisie
        plage_service = st.number_input(
            "Plage de température de service (°C)", min_value=0.0,
            value=_etat("plage", 40.0), step=5.0, disabled=not plage_saisie,
        )
        st.session_state["plage"] = plage_service
    with c3:
        remontage = st.selectbox(
            "Démontage/remontage sur la campagne",
            ["Non précisé", "Oui", "Non"],
            index=["Non précisé", "Oui", "Non"].index(_etat("remontage", "Non précisé")),
            help="« Non » motive la non-calculabilité par l'absence d'objet à mesurer, "
            "plutôt que par une configuration incomplète.",
        )
        st.session_state["remontage"] = remontage

    st.divider()
    a, b = st.columns(2)
    with a:
        st.markdown("**Détection des paliers stabilisés**")
        paliers = {
            "duree_palier_s": st.number_input("Durée minimale d'un palier (s)", 0.2, 60.0, _etat("p_duree", 2.0), 0.5),
            "tolerance_stab_pc_pe": st.number_input("Tolérance de stabilité (% PE)", 0.01, 10.0, _etat("p_tol", 0.5), 0.1),
            "fraction_finale": st.slider("Fraction finale moyennée", 0.1, 1.0, _etat("p_frac", 0.5), 0.05),
            "ecart_min_paliers_pc_pe": st.number_input("Écart minimal entre paliers (% PE)", 0.1, 20.0, _etat("p_ecart", 1.0), 0.1),
            "tolerance_appariement_pc_pe": st.number_input("Tolérance d'appariement (% PE)", 0.1, 20.0, _etat("p_app", 1.0), 0.1),
        }
        st.markdown("**Relevés de zéro**")
        zero = {
            "duree_fenetre_s": st.number_input("Durée de fenêtre de repos (s)", 0.5, 120.0, _etat("z_duree", 5.0), 1.0),
            "seuil_couple_ref_pc_pe": st.number_input("Seuil de couple au repos (% PE)", 0.01, 10.0, _etat("z_couple", 1.0), 0.1),
            "seuil_regime": st.number_input("Seuil de régime au repos", 0.0, 5000.0, _etat("z_regime", 20.0), 5.0),
            "fraction_bord": st.slider("Position exigée des relevés (fraction de l'essai)", 0.05, 0.5, _etat("z_bord", 0.25), 0.05),
        }
    with b:
        st.markdown("**Recalage temporel**")
        intercorrelation = {
            "retard_max_ms": st.number_input("Retard maximal recherché (ms)", 1.0, 5000.0, _etat("i_max", 500.0), 50.0),
            "passe_haut_Hz": st.number_input("Passe-haut avant corrélation (Hz)", 0.0, 50.0, _etat("i_ph", 0.2), 0.1),
            "seuil_activite_pc_pe": st.number_input("Seuil d'activité dynamique (% PE)", 0.1, 50.0, _etat("i_act", 2.0), 0.5),
            "fenetre_activite_s": st.number_input("Fenêtre d'activité (s)", 0.1, 30.0, _etat("i_fen", 1.0), 0.5),
            "duree_comblement_s": st.number_input("Comblement des interruptions (s)", 0.0, 60.0, _etat("i_comb", 5.0), 1.0),
        }
        st.markdown("**Sensibilité thermique**")
        thermique = {
            "amplitude_min_C": st.number_input("Excursion thermique minimale (°C)", 0.5, 100.0, _etat("t_amp", 5.0), 1.0),
            "plage_service_C": float(plage_service) if plage_saisie else None,
        }
        st.markdown("**Seuils de rédaction de la conclusion**")
        diagnostic = {
            "seuil_offset_pc_pe": st.number_input("Offset significatif (% PE)", 0.01, 10.0, _etat("d_off", 0.5), 0.1),
            "seuil_gain_pc": st.number_input("Gain significatif (%)", 0.01, 20.0, _etat("d_gain", 1.0), 0.1),
            "seuil_retard_ms": st.number_input("Retard significatif (ms)", 0.1, 500.0, _etat("d_ret", 5.0), 1.0),
            "gain_rms_recalage": st.slider("Baisse de RMS attribuée à la synchronisation", 0.05, 0.95, _etat("d_rms", 0.30), 0.05),
            "seuil_correlation_point_fct": st.slider("|r| résidu / point de fonctionnement", 0.1, 0.95, _etat("d_r", 0.5), 0.05),
        }
    if intercorrelation["passe_haut_Hz"] == 0.0:
        intercorrelation["passe_haut_Hz"] = None

    st.session_state["params"] = {
        "paliers": paliers, "intercorrelation": intercorrelation,
        "zero": zero, "thermique": thermique, "diagnostic": diagnostic,
    }

# ---------------------------------------------------------------------------
# 5 · Analyse & résultats
# ---------------------------------------------------------------------------


def _configuration_courante() -> dict | None:
    if not st.session_state.get("inventaire"):
        return None
    params = st.session_state.get("params")
    if not params:
        st.warning("Ouvre l'onglet « Hypothèses » une fois pour fixer les paramètres.")
        return None
    remontage_choix = {"Oui": True, "Non": False, "Non précisé": None}[
        st.session_state.get("remontage", "Non précisé")
    ]
    return E.construire_dict(
        racine=st.session_state["racine"],
        dossier_sortie=st.session_state["dossier_sortie"],
        pleine_echelle_Nm=st.session_state["pe"],
        mode_comparaison=st.session_state["mode"],
        rapport_reduction=st.session_state["rapport"],
        incertitude_reference_k1_Nm=(
            float(st.session_state["u_ref"]) if st.session_state.get("u_ref_connue") else None
        ),
        frequence_Hz=st.session_state["freq"],
        remontage_realise=remontage_choix,
        canaux=st.session_state.get("canaux", {}),
        essais=st.session_state.get("essais", {}),
        **params,
    )


with onglets[4]:
    st.subheader("Analyse")
    configuration = _configuration_courante()

    if configuration is None:
        st.info("Complète les onglets 1 à 4.")
    else:
        cfg, erreurs = E.valider(configuration)
        if erreurs:
            st.error("Configuration invalide :\n\n" + "\n".join(f"- {e}" for e in erreurs))
        else:
            for avertissement in cfg.verifier_mapping():
                st.warning(avertissement)
            for avertissement in cfg.verifier_saisies():
                st.info(avertissement)

            c1, c2 = st.columns([1, 2])
            with c1:
                lancer_analyse = st.button("Lancer l'analyse", type="primary",
                                           use_container_width=True)
            with c2:
                st.download_button(
                    "Télécharger la configuration (.yaml)",
                    data=E.vers_yaml(configuration).encode("utf-8"),
                    file_name="correlation.yaml", mime="text/yaml",
                    use_container_width=True,
                    help="Rejoue à l'identique en ligne de commande : "
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
                    st.rerun()  # rafraîchit l'indicateur d'étape de la barre latérale
                except Exception as exc:
                    st.session_state["campagne"] = None
                    st.error(f"Échec de l'analyse : {exc}")

    campagne = st.session_state.get("campagne")
    if campagne is not None:
        pe = campagne.config.pleine_echelle_Nm
        st.divider()

        # -- quatre chiffres clés ------------------------------------------
        balayage = next(
            (e for e in campagne.par_type("balayage")
             if e.regression and not e.regression.non_calculable), None
        )
        reg = balayage.regression if balayage else None
        retards = [
            e.recalage_principal.retard_ms for e in campagne.par_type("dynamique")
            if e.recalage_principal and not e.recalage_principal.non_calculable
        ]
        inc = campagne.incertitude

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Offset b", _valeur(100 * reg.b / pe, 3, True, " % PE") if reg else "non calculable")
        m2.metric("Erreur de sensibilité",
                  _valeur(100 * (reg.a - 1), 3, True, " %") if reg else "non calculable")
        m3.metric("Retard temporel",
                  _valeur(sorted(retards)[len(retards) // 2], 1, True, " ms") if retards else "non calculable")
        m4.metric("Incertitude élargie (k=2)",
                  _valeur(inc.U_k2_pc_pe, 3, False, " % PE") if inc and not inc.non_calculable else "non calculable")
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
        vues = st.tabs(["Tableau récapitulatif", "Conclusions", "Figures",
                        "Incertitude", "Cartes de contrôle", "Détail par essai", "Export"])

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
            texte = R.rediger(campagne)
            st.download_button(
                "Rapport Markdown (.md)", data=texte.encode("utf-8"),
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
