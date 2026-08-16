"""Validation bout en bout sur l'arborescence synthétique à vérité connue.

Le pipeline complet est exécuté (lecture MDF4 → traitements → rapport → PNG) et
l'on vérifie qu'il retrouve les défauts injectés par le générateur.

Ces tests valident aussi les deux garanties structurantes :
  * les `.mf4` sources ne sont pas modifiés (empreinte + date de modification) ;
  * une grandeur sans donnée est déclarée non calculable, jamais estimée.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from amdec_correlation import rapport as R  # noqa: E402
from amdec_correlation.analyse import analyser  # noqa: E402
from amdec_correlation.config import Config  # noqa: E402
from generer_mf4_synthetique import (  # noqa: E402
    BIAIS_REMONTAGE_NM,
    GAIN,
    HYSTERESIS_NM,
    NOMS,
    OFFSET_NM,
    PLEINE_ECHELLE,
    RETARD_S,
    SENSIBILITE_TH_NM_PAR_C,
    ECART_VOIES_NM,
    generer,
)


def _configuration(racine: Path, sortie: Path) -> Config:
    return Config.depuis_dict(
        {
            "racine_donnees": str(racine),
            "dossier_sortie": str(sortie),
            "pleine_echelle_Nm": PLEINE_ECHELLE,
            "comparaison": {
                "mode": "moyenne",
                "rapport_reduction": 1.0,
                "incertitude_reference_k1_Nm": 2.0,
            },
            # Les acquisitions synthétiques sont à 200 Hz sur les voies de couple.
            "acquisition": {"frequence_reechantillonnage_Hz": 200},
            "canaux": {
                "defaut": {
                    "couple_mesure_gauche": NOMS["gauche"],
                    "couple_mesure_droite": NOMS["droite"],
                    "couple_reference_gauche": NOMS["reference_gauche"],
                    "couple_reference_droite": NOMS["reference_droite"],
                    "regime": NOMS["regime"],
                    "temperature": NOMS["temperature"],
                    "etat": [NOMS["etat"]],
                }
            },
            "essais": {
                "1-Balayage Couple": {"type": "balayage", "ligne_droite": True},
                "2-CPC 20°C": {"type": "repetabilite", "groupe_remontage": "avant"},
                "2-CPC 20°C apres remontage": {
                    "type": "repetabilite", "groupe_remontage": "apres"
                },
                "5-Décollage en pente": {"type": "dynamique"},
                "7-WLTC": {"type": "dynamique"},
            },
            "thermique": {"amplitude_min_C": 5.0, "plage_service_C": 40.0},
        }
    )


@pytest.fixture(scope="module")
def campagne(tmp_path_factory):
    racine = tmp_path_factory.mktemp("donnees")
    sortie = tmp_path_factory.mktemp("sortie")
    generer(racine)
    cfg = _configuration(racine, sortie)
    return analyser(cfg), cfg, racine, sortie


# ---------------------------------------------------------------------------
# Lecture seule
# ---------------------------------------------------------------------------


def test_les_sources_mf4_ne_sont_pas_modifiees(tmp_path):
    racine = tmp_path / "donnees"
    generer(racine)
    fichiers = sorted(racine.rglob("*.mf4"))
    assert fichiers, "l'arborescence synthétique doit contenir des .mf4"
    avant = {
        f: (hashlib.sha256(f.read_bytes()).hexdigest(), f.stat().st_mtime_ns) for f in fichiers
    }

    cfg = _configuration(racine, tmp_path / "sortie")
    analyser(cfg)

    for fichier, (empreinte, mtime) in avant.items():
        assert hashlib.sha256(fichier.read_bytes()).hexdigest() == empreinte, (
            f"{fichier.name} a été modifié"
        )
        assert fichier.stat().st_mtime_ns == mtime


# ---------------------------------------------------------------------------
# Grandeurs retrouvées
# ---------------------------------------------------------------------------


def test_balayage_retrouve_gain_et_offset(campagne):
    resultats, cfg, _, _ = campagne
    essai = resultats.essai("1-Balayage Couple")
    reg = essai.regression
    assert reg is not None and reg.non_calculable is None
    # 17 paliers : 9 en montée, 8 en descente.
    assert len(essai.paliers) == 17
    assert reg.a == pytest.approx(GAIN, abs=0.002)
    # L'offset apparent inclut la demi-hystérésis, la régression portant sur les
    # deux sens à la fois.
    assert reg.b == pytest.approx(OFFSET_NM - HYSTERESIS_NM / 2, abs=1.0)


def test_balayage_retrouve_l_hysteresis(campagne):
    resultats, _, _, _ = campagne
    hyst = resultats.essai("1-Balayage Couple").hysteresis
    assert hyst.non_calculable is None
    assert hyst.max_Nm == pytest.approx(HYSTERESIS_NM, abs=0.75)
    assert hyst.n_appariements >= 7


def test_non_linearite_reste_faible_sur_un_capteur_lineaire(campagne):
    resultats, _, _, _ = campagne
    essai = resultats.essai("1-Balayage Couple")
    # Le générateur n'injecte aucune non-linéarité : seuls le bruit et la
    # demi-hystérésis subsistent, soit très en deçà de 7,5 N·m.
    assert essai.non_linearite_Nm < 7.5


def test_repetabilite_est_calculee_sur_les_essais_repetes(campagne):
    resultats, _, _, _ = campagne
    rep = resultats.essai("2-CPC 20°C").repetabilite
    assert rep.non_calculable is None
    assert rep.degres_liberte > 0
    # Biais injectés 0 / +1,2 / −0,9 N·m → σ ≈ 1,05 N·m.
    assert rep.ecart_type_Nm == pytest.approx(1.05, abs=0.45)


def test_repetabilite_apres_remontage_retrouve_le_biais(campagne):
    resultats, _, _, _ = campagne
    rep = resultats.repetabilite_remontage
    assert resultats.motif_remontage is None
    assert rep is not None and rep.non_calculable is None
    # Deux groupes séparés de ~BIAIS_REMONTAGE_NM : l'écart-type de deux valeurs
    # séparées de d vaut d/√2.
    attendu_Nm = (BIAIS_REMONTAGE_NM + 0.5) / 2**0.5
    assert rep.ecart_type_Nm == pytest.approx(attendu_Nm, rel=0.35)


@pytest.mark.parametrize("essai", ["7-WLTC", "5-Décollage en pente"])
def test_recalage_retrouve_le_retard_injecte(campagne, essai):
    resultats, _, _, _ = campagne
    recal = resultats.essai(essai).recalage_principal
    assert recal is not None and recal.non_calculable is None
    assert recal.retard_ms == pytest.approx(RETARD_S * 1000.0, abs=4.0)
    assert recal.rms_residu_apres_Nm < recal.rms_residu_avant_Nm


def test_sensibilite_thermique_retrouve_la_pente(campagne):
    resultats, _, _, _ = campagne
    th = resultats.thermique_globale
    assert th is not None and th.non_calculable is None
    assert th.pente_Nm_par_C == pytest.approx(SENSIBILITE_TH_NM_PAR_C, rel=0.12)


def test_sensibilite_thermique_separe_l_effet_du_couple(campagne):
    """L'erreur de gain ne doit pas être comptée comme un effet thermique.

    Couple et température montent ensemble au fil du cycle : sans régression
    multiple, une part de l'erreur de gain serait attribuée à la température.
    Le coefficient du couple doit ressortir égal à l'erreur de gain injectée.
    """
    resultats, _, _, _ = campagne
    th = resultats.thermique_globale
    assert th.correction_couple is True
    assert th.coefficient_couple == pytest.approx(GAIN - 1.0, rel=0.15)
    assert th.r2 > 0.9


def test_derive_zero_coherente_avec_la_derive_thermique(campagne):
    resultats, _, _, _ = campagne
    derives = resultats.essai("7-WLTC").derives_zero_valides
    assert derives, "les zéros de début et de fin du WLTC doivent être détectés"
    # Les deux fenêtres de repos sont à ~21 °C et ~44 °C : la dérive attendue
    # vaut la sensibilité thermique multipliée par cet écart.
    attendu_Nm = SENSIBILITE_TH_NM_PAR_C * 23.0
    assert derives[0].derive_Nm == pytest.approx(attendu_Nm, rel=0.3)


def test_redondance_retrouve_l_ecart_entre_voies(campagne):
    resultats, _, _, _ = campagne
    red = resultats.essai("1-Balayage Couple").redondance
    assert red is not None and red.non_calculable is None
    assert red.moyenne_Nm == pytest.approx(ECART_VOIES_NM, abs=0.3)


def test_bilan_incertitude_est_complet_quand_tout_est_fourni(campagne):
    resultats, _, _, _ = campagne
    inc = resultats.incertitude
    assert inc.non_calculable is None
    assert inc.U_k2_Nm > 0
    # Toutes les contributions sont disponibles ici : le résultat n'est pas un minorant.
    assert not inc.exclusions, f"exclusions inattendues : {inc.exclusions}"
    assert inc.minorant is False


def test_parametres_spc_sont_issus_des_essais_repetes(campagne):
    resultats, _, _, _ = campagne
    spc = resultats.spc
    assert spc.non_calculable is None
    assert spc.sigma0_Nm > 0
    assert spc.cusum_h_alarme_Nm == pytest.approx(5.0 * spc.sigma0_Nm)
    assert spc.ewma_limite_alarme_Nm > spc.ewma_limite_alerte_Nm


def test_diagnostic_dynamique_designe_la_cause_dominante(campagne):
    """Deux défauts coexistent dans les données : le diagnostic doit trancher.

    Le générateur injecte à la fois une erreur de gain de +1,2 % et un retard de
    40 ms. Le recalage temporel ne réduit le résidu que d'environ 15 % : l'écart
    n'est donc PAS « uniquement en transitoire » et la grille doit conclure au
    gain — tout en signalant le retard résiduel dans la conclusion rédigée.
    """
    resultats, _, _, _ = campagne
    diag = resultats.essai("7-WLTC").diagnostic
    assert diag is not None
    assert diag.cause == "gain"
    assert "recalage" in diag.conclusion and "ms" in diag.conclusion
    phrases = [p for p in diag.conclusion.split(". ") if p.strip()]
    assert len(phrases) <= 3


def test_diagnostic_conclut_a_la_synchronisation_si_le_retard_domine(campagne):
    """Sans erreur de gain, le même retard doit être attribué à la synchronisation."""
    from amdec_correlation import metriques as M

    resultats, cfg, _, _ = campagne
    essai = resultats.essai("7-WLTC")
    x = __import__("numpy").linspace(0, 1200, 30)
    reg_sans_defaut = M.regression(x, x)  # a = 1, b = 0
    recal = essai.recalage_principal
    # Résidu presque entièrement supprimé par le recalage.
    recal_domine = M.Recalage(
        retard_ms=recal.retard_ms, correlation_pic=recal.correlation_pic,
        rms_residu_avant_Nm=40.0, rms_residu_apres_Nm=4.0,
    )
    diag = M.diagnostiquer(
        "l'essai « 7-WLTC »", reg_sans_defaut, recal_domine,
        essai.r_residu_regime, essai.r_residu_couple,
        cfg.pleine_echelle_Nm, cfg.diagnostic,
    )
    assert diag.cause == "synchronisation"


# ---------------------------------------------------------------------------
# Restitutions
# ---------------------------------------------------------------------------


def test_le_rapport_contient_toutes_les_lignes_attendues(campagne):
    resultats, _, _, sortie = campagne
    chemin = R.ecrire(resultats)
    texte = chemin.read_text(encoding="utf-8")
    for ligne in (
        "Offset b (N·m)",
        "Erreur de sensibilité a−1 (%)",
        "Non-linéarité (N·m)",
        "Hystérésis (N·m)",
        "Répétabilité après remontage (N·m)",
        "Retard temporel (ms)",
        "Sensibilité thermique (N·m pour 10 °C)",
        "Dérive de zéro sur cycle (N·m)",
        "Incertitude élargie résultante (k=2)",
    ):
        assert ligne in texte, f"ligne absente du tableau : {ligne}"
    assert "Hypothèses de traitement" in texte
    assert "cartes de contrôle" in texte


def test_les_figures_sont_generees(campagne):
    resultats, cfg, _, sortie = campagne
    figures = resultats.figures + [f for e in resultats.essais for f in e.figures]
    assert figures, "aucune figure produite"
    for figure in figures:
        assert figure.exists() and figure.stat().st_size > 5000
    noms = {f.name for f in figures}
    assert "residu_vs_temperature.png" in noms
    assert any("regression" in n for n in noms)
    assert any("recalage" in n for n in noms)


# ---------------------------------------------------------------------------
# Non-calculabilité explicite
# ---------------------------------------------------------------------------


def test_canal_absent_rend_la_grandeur_non_calculable_sans_l_estimer(tmp_path):
    """Sans canal de température, la sensibilité thermique doit être refusée."""
    racine = tmp_path / "donnees"
    generer(racine)
    cfg = _configuration(racine, tmp_path / "sortie")
    cfg._canaux_defaut["temperature"] = None

    resultats = analyser(cfg)
    th = resultats.thermique_globale
    assert th.non_calculable is not None
    assert "température" in th.non_calculable

    texte = R.rediger(resultats)
    assert "non calculable" in texte
    # La contribution thermique doit être explicitement exclue du bilan.
    assert resultats.incertitude.minorant is True


def test_un_seul_groupe_de_remontage_rend_la_grandeur_non_calculable(tmp_path):
    racine = tmp_path / "donnees"
    generer(racine)
    cfg = _configuration(racine, tmp_path / "sortie")
    cfg.essais = tuple(
        e for e in cfg.essais if e.dossier != "2-CPC 20°C apres remontage"
    )

    resultats = analyser(cfg)
    assert resultats.repetabilite_remontage is None
    assert "groupe(s) de remontage" in resultats.motif_remontage
    texte = R.rediger(resultats)
    assert "Répétabilité après remontage (N·m) | **non calculable**" in texte


def _pente_mesure_reference(donnees) -> float:
    """Pente de mesuré vs référence.

    C'est la bonne grandeur pour comparer deux modes : un rapport moyen
    `mesure / référence` serait dominé par l'offset aux faibles couples.
    """
    return float(np.polyfit(donnees.reference, donnees.mesure, 1)[0])


def test_le_mode_de_comparaison_s_applique_aux_deux_cotes(tmp_path):
    """Mesuré et référence doivent être combinés de la même façon.

    Le banc fournit deux voies de référence. En mode « somme », la référence
    doit valoir refG + refD — sinon on comparerait une somme de voies mesurées à
    une moyenne de voies de référence, et l'erreur de sensibilité afficherait un
    facteur 2.
    """
    from amdec_correlation.analyse import preparer

    racine = tmp_path / "donnees"
    generer(racine)
    fichier = next((racine / "1-Balayage Couple").glob("*.mf4"))

    pentes = {}
    for mode in ("moyenne", "somme", "gauche"):
        cfg = _configuration(racine, tmp_path / "sortie")
        cfg.mode_comparaison = mode
        pentes[mode] = _pente_mesure_reference(
            preparer(fichier, cfg, cfg.canaux_pour("1-Balayage Couple"))
        )

    assert pentes["somme"] == pytest.approx(pentes["moyenne"], rel=1e-6)
    assert pentes["gauche"] == pytest.approx(pentes["moyenne"], rel=1e-3)
    assert pentes["moyenne"] == pytest.approx(GAIN, abs=0.002)


def test_une_reference_en_voie_unique_reste_acceptee(tmp_path):
    """Repli documenté : un banc ne fournissant qu'une voie reste exploitable."""
    from amdec_correlation.analyse import preparer

    racine = tmp_path / "donnees"
    generer(racine)
    cfg = _configuration(racine, tmp_path / "sortie")
    cfg._canaux_defaut["couple_reference"] = NOMS["reference_gauche"]
    cfg._canaux_defaut["couple_reference_gauche"] = None
    cfg._canaux_defaut["couple_reference_droite"] = None

    fichier = next((racine / "1-Balayage Couple").glob("*.mf4"))
    d = preparer(fichier, cfg, cfg.canaux_pour("1-Balayage Couple"))
    assert _pente_mesure_reference(d) == pytest.approx(GAIN, abs=0.002)


def test_les_saisies_utilisateur_manquantes_sont_signalees(tmp_path):
    """L'incertitude du banc et la plage de service sont des saisies utilisateur.

    Elles ne peuvent pas être déduites des acquisitions : leur absence doit être
    visible dès l'exécution, et le bilan doit s'annoncer comme un minorant.
    """
    racine = tmp_path / "donnees"
    generer(racine)
    cfg = _configuration(racine, tmp_path / "sortie")
    cfg.incertitude_reference_k1_Nm = None
    object.__setattr__(cfg.thermique, "plage_service_C", None)

    saisies = cfg.verifier_saisies()
    assert any("incertitude_reference_k1_Nm" in a for a in saisies)
    assert any("plage_service_C" in a for a in saisies)

    resultats = analyser(cfg)
    assert resultats.incertitude.minorant is True
    assert any("incertitude du moyen de référence" in e for e in resultats.incertitude.exclusions)
    texte = R.rediger(resultats)
    assert "MINORANT" in texte or "minorant" in texte


def test_aucune_saisie_signalee_quand_tout_est_renseigne(campagne):
    resultats, cfg, _, _ = campagne
    assert cfg.verifier_saisies() == []
    assert resultats.incertitude.minorant is False


def test_campagne_sans_remontage_est_motivee_comme_non_definie(tmp_path):
    """Une grandeur non DÉFINIE ne doit pas être motivée comme non CONFIGURÉE.

    Sans remontage dans la campagne, le rapport doit dire qu'il n'y a rien à
    mesurer — pas qu'une clé de configuration manque, ce qui se lirait comme un
    oubli de l'analyste.
    """
    racine = tmp_path / "donnees"
    generer(racine)
    cfg = _configuration(racine, tmp_path / "sortie")
    cfg.essais = tuple(
        e for e in cfg.essais if e.dossier != "2-CPC 20°C apres remontage"
    )
    cfg.remontage_realise = False

    resultats = analyser(cfg)
    assert resultats.repetabilite_remontage is None
    motif = resultats.motif_remontage
    assert "aucun démontage ni remontage" in motif
    assert "configuration" not in motif
    texte = R.rediger(resultats)
    assert "Répétabilité après remontage (N·m) | **non calculable**" in texte
    # La répétabilité sans remontage doit rester citée en repère, distinguée.
    assert "ce n'est pas la même grandeur" in texte


def test_le_drapeau_ne_prime_pas_sur_des_donnees_de_remontage_reelles(tmp_path):
    """Si deux groupes sont réellement déclarés, la grandeur se calcule."""
    racine = tmp_path / "donnees"
    generer(racine)
    cfg = _configuration(racine, tmp_path / "sortie")
    cfg.remontage_realise = False  # contredit par les groupes déclarés

    resultats = analyser(cfg)
    assert resultats.motif_remontage is None
    assert resultats.repetabilite_remontage is not None
    assert resultats.repetabilite_remontage.non_calculable is None
