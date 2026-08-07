"""Validation des formules sur signaux à vérité connue.

Ces tests ne valident pas « le code tourne » mais « le code retrouve la bonne
valeur » : chaque cas injecte un défaut d'amplitude connue et vérifie que la
grandeur restituée le retrouve, y compris dans son signe.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amdec_correlation import metriques as M  # noqa: E402
from amdec_correlation.config import (  # noqa: E402
    ParamsDiagnostic,
    ParamsIntercorrelation,
    ParamsPaliers,
    ParamsThermique,
    ParamsZero,
)

PE = 1500.0


# ---------------------------------------------------------------------------
# Régression
# ---------------------------------------------------------------------------


def test_regression_retrouve_gain_et_offset():
    x = np.linspace(0, 1200, 40)
    y = 1.012 * x + 6.0
    reg = M.regression(x, y)
    assert reg.non_calculable is None
    assert reg.a == pytest.approx(1.012, abs=1e-9)
    assert reg.b == pytest.approx(6.0, abs=1e-9)
    assert reg.r2 == pytest.approx(1.0, abs=1e-12)


def test_regression_refuse_variable_constante():
    reg = M.regression(np.full(20, 300.0), np.arange(20.0))
    assert reg.non_calculable is not None
    assert "constante" in reg.non_calculable


def test_regression_refuse_trop_peu_de_points():
    reg = M.regression([1.0, 2.0], [1.0, 2.0])
    assert reg.non_calculable is not None


# ---------------------------------------------------------------------------
# Paliers, hystérésis, non-linéarité
# ---------------------------------------------------------------------------


def _escalier(niveaux, fe=100.0, duree_palier=5.0, duree_rampe=1.0):
    t_liste, ref_liste = [], []
    horloge = 0.0
    for i, niveau in enumerate(niveaux):
        n = int(duree_palier * fe)
        t_liste.append(horloge + np.arange(n) / fe)
        ref_liste.append(np.full(n, float(niveau)))
        horloge += duree_palier
        if i < len(niveaux) - 1:
            m = int(duree_rampe * fe)
            t_liste.append(horloge + np.arange(m) / fe)
            ref_liste.append(np.linspace(niveaux[i], niveaux[i + 1], m))
            horloge += duree_rampe
    return np.concatenate(t_liste), np.concatenate(ref_liste)


def test_detection_paliers_et_sens():
    niveaux = [0, 300, 600, 900, 600, 300, 0]
    t, ref = _escalier(niveaux)
    paliers = M.detecter_paliers(t, ref, ref.copy(), PE, ParamsPaliers())

    assert len(paliers) == len(niveaux)
    for palier, attendu in zip(paliers, niveaux):
        assert palier.reference == pytest.approx(attendu, abs=1.0)
    # Les trois premiers montent, les trois derniers descendent.
    assert [p.sens for p in paliers[:4]] == ["montee"] * 4
    assert [p.sens for p in paliers[4:]] == ["descente"] * 3


def test_hysteresis_retrouve_ecart_injecte():
    niveaux = [0, 300, 600, 900, 600, 300, 0]
    t, ref = _escalier(niveaux)
    # 3 N·m d'écart appliqué uniquement à la descente, soit 0,2 % PE.
    montee = np.ones_like(ref, dtype=bool)
    montee[np.searchsorted(t, t[np.argmax(ref)]) :] = False
    mesure = ref.copy()
    mesure[~montee] -= 3.0

    paliers = M.detecter_paliers(t, ref, mesure, PE, ParamsPaliers())
    hyst = M.hysteresis(paliers, PE, ParamsPaliers())
    assert hyst.non_calculable is None
    assert hyst.max_pc_pe == pytest.approx(100 * 3.0 / PE, abs=0.01)


def test_hysteresis_non_calculable_sans_descente():
    t, ref = _escalier([0, 300, 600, 900])
    paliers = M.detecter_paliers(t, ref, ref.copy(), PE, ParamsPaliers())
    hyst = M.hysteresis(paliers, PE, ParamsPaliers())
    assert hyst.non_calculable is not None
    assert "descendante" in hyst.non_calculable


def test_non_linearite_retrouve_ecart_maximal():
    x = np.linspace(0, 1200, 25)
    y = x.copy()
    y[12] += 7.5  # écart isolé de 7,5 N·m, soit 0,5 % PE
    reg = M.regression(x, y)
    assert M.non_linearite_pc_pe(reg, PE) == pytest.approx(0.5, rel=0.15)


# ---------------------------------------------------------------------------
# Répétabilité
# ---------------------------------------------------------------------------


def test_repetabilite_retrouve_dispersion_injectee():
    biais = [0.0, 3.0, -3.0, 1.5, -1.5]  # écart-type exact = 2.4109...
    paliers = [
        M.Palier(0, 1, reference=niveau, mesure=niveau + b, ecart_type_mesure=0,
                 ecart_type_reference=0, n=100, source=f"f{i}")
        for i, b in enumerate(biais)
        for niveau in (200.0, 600.0, 1000.0)
    ]
    rep = M.repetabilite(paliers, PE, ParamsPaliers())
    assert rep.non_calculable is None
    attendu = float(np.std(biais, ddof=1))
    assert rep.ecart_type_Nm == pytest.approx(attendu, rel=1e-9)
    assert rep.degres_liberte == 3 * (len(biais) - 1)


def test_repetabilite_non_calculable_sans_repetition():
    paliers = [
        M.Palier(0, 1, reference=n, mesure=n, ecart_type_mesure=0,
                 ecart_type_reference=0, n=10)
        for n in (200.0, 600.0, 1000.0)
    ]
    rep = M.repetabilite(paliers, PE, ParamsPaliers())
    assert rep.non_calculable is not None
    assert "répété" in rep.non_calculable


# ---------------------------------------------------------------------------
# Recalage temporel
# ---------------------------------------------------------------------------


def _signal_dynamique(duree=60.0, fe=200.0, graine=0):
    rng = np.random.default_rng(graine)
    t = np.arange(0, duree, 1 / fe)
    ref = (
        400.0
        + 300.0 * np.sin(2 * np.pi * t / 11.0)
        + 180.0 * np.sin(2 * np.pi * t / 3.7 + 0.8)
        + 80.0 * np.sin(2 * np.pi * t / 1.9 + 0.2)
        + rng.normal(0, 2.0, t.size)
    )
    return t, ref


@pytest.mark.parametrize("retard_s", [0.040, 0.120, -0.075])
def test_recalage_retrouve_retard_et_son_signe(retard_s):
    t, ref = _signal_dynamique()
    # mesure(t) = ref(t − retard) : la mesure est EN RETARD de `retard_s`.
    mesure = np.interp(t - retard_s, t, ref, left=ref[0], right=ref[-1])

    recal = M.recalage_temporel(t, ref, mesure, PE, ParamsIntercorrelation())
    assert recal.non_calculable is None
    assert recal.retard_ms == pytest.approx(retard_s * 1000.0, abs=3.0)
    assert recal.correlation_pic > 0.9
    # Le recalage doit réduire nettement le résidu.
    assert recal.rms_residu_apres_Nm < 0.25 * recal.rms_residu_avant_Nm


def test_recalage_non_calculable_sur_signal_stationnaire():
    t = np.arange(0, 60, 1 / 200.0)
    ref = np.full(t.size, 500.0)
    recal = M.recalage_temporel(t, ref, ref.copy(), PE, ParamsIntercorrelation())
    assert recal.non_calculable is not None
    assert "dynamique" in recal.non_calculable


def test_appliquer_retard_est_l_inverse_du_decalage():
    t, ref = _signal_dynamique(duree=30.0)
    mesure = np.interp(t - 0.05, t, ref, left=ref[0], right=ref[-1])
    recalee = M.appliquer_retard(t, mesure, 0.05)
    valides = np.isfinite(recalee)
    # Hors bords, le signal recalé retombe sur la référence.
    interieur = valides & (t > 1.0) & (t < 29.0)
    assert np.max(np.abs(recalee[interieur] - ref[interieur])) < 1e-6


# ---------------------------------------------------------------------------
# Sensibilité thermique
# ---------------------------------------------------------------------------


def test_sensibilite_thermique_retrouve_la_pente():
    T = np.linspace(20.0, 60.0, 4000)
    residu = 0.15 * (T - 20.0) + np.random.default_rng(0).normal(0, 0.3, T.size)
    sens = M.sensibilite_thermique(T, residu, PE, ParamsThermique())
    assert sens.non_calculable is None
    assert sens.pente_Nm_par_C == pytest.approx(0.15, rel=0.05)
    assert sens.pc_pe_pour_10C == pytest.approx(100 * 0.15 * 10 / PE, rel=0.05)


def test_sensibilite_thermique_separe_couple_et_temperature():
    """Sans découplage, le gain contaminerait la pente thermique.

    Couple et température sont ici volontairement corrélés (l'échauffement suit
    la sollicitation), comme sur un cycle réel.
    """
    rng = np.random.default_rng(0)
    n = 20000
    couple = rng.uniform(0, 1200, n)
    temperature = 20.0 + 0.02 * couple + rng.normal(0, 2.0, n)  # corrélé au couple
    residu = 0.012 * couple + 0.15 * (temperature - 20.0) + rng.normal(0, 0.5, n)

    naif = M.sensibilite_thermique(temperature, residu, PE, ParamsThermique())
    decouple = M.sensibilite_thermique(temperature, residu, PE, ParamsThermique(),
                                       couple=couple)

    assert decouple.correction_couple is True
    assert decouple.pente_Nm_par_C == pytest.approx(0.15, rel=0.1)
    assert decouple.coefficient_couple == pytest.approx(0.012, rel=0.1)
    # La régression naïve surestime nettement l'effet thermique.
    assert naif.pente_Nm_par_C > 1.5 * decouple.pente_Nm_par_C


def test_sensibilite_thermique_non_calculable_si_excursion_trop_faible():
    T = np.linspace(20.0, 22.0, 500)  # 2 °C seulement
    sens = M.sensibilite_thermique(T, 0.15 * T, PE, ParamsThermique(amplitude_min_C=5.0))
    assert sens.non_calculable is not None
    assert "excursion thermique" in sens.non_calculable


# ---------------------------------------------------------------------------
# Dérive du zéro
# ---------------------------------------------------------------------------


def test_derive_zero_retrouve_la_derive():
    fe = 100.0
    t = np.arange(0, 200.0, 1 / fe)
    ref = np.where((t > 20) & (t < 180), 600.0, 0.0)
    regime = np.where((t > 20) & (t < 180), 900.0, 0.0)
    # Zéro à 0 N·m au début, à 4,5 N·m à la fin : dérive de +0,3 % PE.
    mesure = ref.copy() + np.clip((t - 20.0) / 160.0, 0, 1) * 4.5
    derive = M.derive_zero(t, mesure, ref, regime, PE, ParamsZero())
    assert derive.non_calculable is None
    assert derive.derive_pc_pe == pytest.approx(0.3, abs=0.02)


def test_derive_zero_non_calculable_sans_repos_final():
    fe = 100.0
    t = np.arange(0, 200.0, 1 / fe)
    ref = np.where(t > 20, 600.0, 0.0)  # pas de retour au repos
    derive = M.derive_zero(t, ref.copy(), ref, None, PE, ParamsZero())
    assert derive.non_calculable is not None


# ---------------------------------------------------------------------------
# Redondance et incertitude
# ---------------------------------------------------------------------------


def test_redondance_retrouve_ecart_moyen():
    g = np.full(1000, 500.0)
    d = np.full(1000, 498.0)
    red = M.redondance_gauche_droite(g, d, PE)
    assert red.moyenne_Nm == pytest.approx(2.0)
    assert red.moyenne_pc_pe == pytest.approx(100 * 2.0 / PE)


def test_redondance_non_calculable_avec_une_seule_voie():
    red = M.redondance_gauche_droite(np.zeros(10), None, PE)
    assert red.non_calculable is not None


def test_bilan_incertitude_somme_quadratique():
    contributions = [
        M.Contribution("A", 3.0, "normale"),
        M.Contribution("B", 4.0, "normale"),
    ]
    bilan = M.bilan_incertitude(contributions, [], PE)
    assert bilan.u_composee_Nm == pytest.approx(5.0)
    assert bilan.U_k2_Nm == pytest.approx(10.0)
    assert bilan.minorant is False


def test_bilan_incertitude_signale_le_minorant():
    bilan = M.bilan_incertitude(
        [M.Contribution("A", 3.0, "normale")], ["référence banc non fournie"], PE
    )
    assert bilan.minorant is True
    assert bilan.exclusions


def test_loi_rectangulaire_divise_par_racine_de_trois():
    c = M.Contribution("NL", 9.0, "rectangulaire")
    assert c.u_Nm == pytest.approx(9.0 / np.sqrt(3))


# ---------------------------------------------------------------------------
# Cartes de contrôle
# ---------------------------------------------------------------------------


def test_parametres_spc_derivent_de_la_dispersion_mesuree():
    rng = np.random.default_rng(0)
    echantillons = rng.normal(0.4, 0.05, 500)
    spc = M.parametres_spc(echantillons)
    assert spc.non_calculable is None
    assert spc.mu0_pc_pe == pytest.approx(0.4, abs=0.01)
    assert spc.sigma0_pc_pe == pytest.approx(0.05, rel=0.1)
    # Les multiplicateurs proviennent des tables ARL, pas d'un choix arbitraire.
    assert spc.cusum_k_pc_pe == pytest.approx(0.5 * spc.sigma0_pc_pe)
    assert spc.cusum_h_alarme_pc_pe == pytest.approx(5.0 * spc.sigma0_pc_pe)
    attendu = spc.sigma0_pc_pe * np.sqrt(0.2 / 1.8)
    assert spc.ewma_sigma_asymptotique_pc_pe == pytest.approx(attendu)


def test_parametres_spc_non_calculables_sans_repetition():
    spc = M.parametres_spc([0.4])
    assert spc.non_calculable is not None
    assert "répétés" in spc.non_calculable


# ---------------------------------------------------------------------------
# Grille de diagnostic
# ---------------------------------------------------------------------------


def _regression_factice(a: float, b: float) -> M.Regression:
    x = np.linspace(0, 1200, 30)
    return M.regression(x, a * x + b)


def test_diagnostic_identifie_la_synchronisation():
    recal = M.Recalage(retard_ms=45.0, correlation_pic=0.99,
                       rms_residu_avant_Nm=40.0, rms_residu_apres_Nm=5.0)
    diag = M.diagnostiquer("l'essai X", _regression_factice(1.0, 0.0), recal,
                           0.1, 0.1, PE, ParamsDiagnostic())
    assert diag.cause == "synchronisation"


def test_diagnostic_identifie_le_gain():
    diag = M.diagnostiquer("l'essai X", _regression_factice(1.03, 0.0), None,
                           0.1, 0.1, PE, ParamsDiagnostic())
    assert diag.cause == "gain"


def test_diagnostic_identifie_l_offset():
    diag = M.diagnostiquer("l'essai X", _regression_factice(1.0, 30.0), None,
                           0.1, 0.1, PE, ParamsDiagnostic())
    assert diag.cause == "offset"


def test_diagnostic_identifie_l_ecart_de_modele():
    diag = M.diagnostiquer("l'essai X", _regression_factice(1.0, 0.0), None,
                           0.85, 0.2, PE, ParamsDiagnostic(), reproductible=True)
    assert diag.cause == "modele"


def test_diagnostic_conclut_conforme_sinon():
    diag = M.diagnostiquer("l'essai X", _regression_factice(1.0, 0.0), None,
                           0.05, 0.05, PE, ParamsDiagnostic())
    assert diag.cause == "conforme"


def test_conclusions_tiennent_en_trois_phrases():
    """La grille impose des conclusions directement collables dans le rapport."""
    for reg, recal, r in (
        (_regression_factice(1.03, 0.0), None, 0.1),
        (_regression_factice(1.0, 30.0), None, 0.1),
        (_regression_factice(1.0, 0.0), M.Recalage(45.0, 0.99, 40.0, 5.0), 0.1),
        (_regression_factice(1.0, 0.0), None, 0.85),
        (_regression_factice(1.0, 0.0), None, 0.05),
    ):
        diag = M.diagnostiquer("l'essai X", reg, recal, r, 0.1, PE, ParamsDiagnostic())
        phrases = [p for p in diag.conclusion.split(". ") if p.strip()]
        assert len(phrases) <= 3, f"{diag.cause} : {len(phrases)} phrases"
