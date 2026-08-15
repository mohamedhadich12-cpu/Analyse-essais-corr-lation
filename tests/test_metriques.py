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


def test_deux_passages_au_meme_niveau_ne_font_pas_un_palier_geant():
    """Un zéro en milieu d'essai et le zéro final restent deux paliers distincts.

    La fusion sert à recoller un palier qu'une perturbation a **momentanément**
    interrompu. Sans borne sur la durée de l'interruption, deux relevés de zéro
    séparés par tout un cycle transitoire n'en formaient qu'un : la figure des
    zones peignait alors une bande de palier par-dessus toute la partie
    dynamique, et la régression recevait un point qui est la moyenne de deux
    mesures faites à des températures différentes.
    """
    fe = 100.0
    t = np.arange(0.0, 200.0, 1.0 / fe)
    reference = np.zeros_like(t)
    # Entre les deux passages à zéro, un cycle franchement dynamique.
    milieu = (t > 20.0) & (t < 180.0)
    reference[milieu] = 500.0 * np.sin(2 * np.pi * t[milieu] / 6.0)

    paliers = M.detecter_paliers(t, reference, reference.copy(), PE, ParamsPaliers())

    zeros = [p for p in paliers if abs(p.reference) < 5.0]
    assert len(zeros) == 2, [f"{p.t_debut:.0f}–{p.t_fin:.0f}" for p in paliers]
    assert zeros[0].t_fin < 25.0
    assert zeros[1].t_debut > 175.0
    assert all(p.t_fin - p.t_debut < 30.0 for p in paliers), (
        "aucun palier ne doit couvrir le cycle qui sépare les deux zéros"
    )


def test_une_interruption_breve_recolle_bien_le_palier():
    """Le cas que la fusion doit continuer de traiter : un à-coup passager."""
    fe = 100.0
    t = np.arange(0.0, 30.0, 1.0 / fe)
    reference = np.full_like(t, 600.0)
    # Un à-coup d'une demi-seconde au milieu d'un palier de 30 s.
    accroc = (t > 14.8) & (t < 15.3)
    reference[accroc] = 700.0

    paliers = M.detecter_paliers(t, reference, reference.copy(), PE, ParamsPaliers())

    assert len(paliers) == 1, [f"{p.t_debut:.1f}–{p.t_fin:.1f}" for p in paliers]
    assert paliers[0].reference == pytest.approx(600.0, abs=5.0)


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


def _forme_depart_arrete(u: np.ndarray, ondulation_retro: float = 80.0) -> np.ndarray:
    """Allure d'un départ arrêté : front raide, décroissance lente, rétro ondulé.

    Le palier de rétro dure trois fois plus longtemps que le transitoire : c'est
    le piège que le choix de plage doit éviter.
    """
    y = np.zeros_like(u)
    front = (u >= 5.5) & (u < 6.2)
    y[front] = 2250.0 * (u[front] - 5.5) / 0.7
    plateau = (u >= 6.2) & (u < 10.5)
    y[plateau] = 2230.0
    decroissance = (u >= 10.5) & (u < 35.5)
    y[decroissance] = 650.0 + 1580.0 * np.exp(-(u[decroissance] - 10.5) / 9.0)
    retro = (u >= 35.5) & (u < 67.0)
    y[retro] = -130.0 + ondulation_retro * np.sin(2 * np.pi * (u[retro] - 35.5) / 1.7)
    return y


def _deux_voies(forme, retard_s: float, graine: int = 0, bruit: float = 6.0):
    """Référence et mesure, avec des bruits INDÉPENDANTS comme deux vraies voies.

    Partager le bruit entre les deux voies fausserait tout : il constituerait un
    repère temporel parfait, et l'intercorrélation retrouverait le retard même
    quand la forme du signal ne le permet pas.
    """
    t = np.arange(0, 97.0, 1 / 200.0)
    rng = np.random.default_rng(graine)
    reference = forme(t) + rng.normal(0, bruit, t.size)
    mesure = forme(t - retard_s) + rng.normal(0, bruit, t.size)
    return t, reference, mesure


def test_la_plage_dynamique_retient_le_transitoire_pas_le_palier_le_plus_long():
    """Le choix se fait sur l'information portée, jamais sur la seule durée."""
    t, ref, _ = _deux_voies(_forme_depart_arrete, 0.0)
    plages = M.plages_dynamiques(t, ref, PE, ParamsIntercorrelation())
    assert len(plages) >= 2, "le palier de rétro doit bien être un candidat"

    debut, fin = plages[0]
    # Le front de couple est vers 5,5 s ; le palier de rétro commence à 35,5 s.
    assert t[debut] < 10.0 and t[fin - 1] < 35.0, (
        f"plage retenue {t[debut]:.1f}–{t[fin - 1]:.1f} s : c'est le palier de rétro, "
        "pas le transitoire"
    )
    # …et ce, alors même que l'autre candidate est bien plus longue.
    ecartee = plages[1]
    assert (ecartee[1] - ecartee[0]) > (fin - debut)


@pytest.mark.parametrize("retard_s", [0.040, -0.040, 0.100])
def test_le_retard_reste_juste_sur_un_depart_arrete(retard_s):
    """Fenêtre courte : la coupure du passe-haut doit s'y adapter.

    Sans cette adaptation, le régime transitoire du filtre occupe une part
    notable de la fenêtre et déplace le pic de plus de dix millisecondes.
    """
    t, ref, mesure = _deux_voies(_forme_depart_arrete, retard_s)
    recal = M.recalage_temporel(t, ref, mesure, PE, ParamsIntercorrelation())
    assert recal.non_calculable is None
    assert recal.retard_ms == pytest.approx(retard_s * 1000.0, abs=8.0)
    assert recal.fenetre[0] < 10.0
    # La coupure a bien été relevée : cinq périodes au moins dans la fenêtre.
    assert recal.coupure_passe_haut_Hz > 0.2
    periodes = recal.coupure_passe_haut_Hz * recal.duree_fenetre_s
    assert periodes >= 0.95 * M.PERIODES_MIN_DANS_FENETRE


def _quatre_departs_arretes(u: np.ndarray) -> np.ndarray:
    """Quatre départs arrêtés dans un même relevé, comme sur banc.

    Chaque front mesure le même retard : les quatre doivent être exploités.
    """
    y = np.zeros_like(u)
    for depart in (10.0, 150.0, 290.0, 430.0):
        front = (u >= depart) & (u < depart + 0.7)
        y[front] = 1000.0 * (u[front] - depart) / 0.7
        # Décroissance linéaire sur 8 s : c'est elle qui rend le départ
        # « actif » assez longtemps pour être corrélé, la pente restant
        # au-dessus du seuil au lieu de s'effondrer comme une exponentielle.
        decroissance = (u >= depart + 0.7) & (u < depart + 8.7)
        y[decroissance] = 1000.0 * (1.0 - (u[decroissance] - depart - 0.7) / 8.0)
        retro = (u >= depart + 8.7) & (u < depart + 120.0)
        y[retro] = -110.0
    return y


def test_toutes_les_plages_dynamiques_sont_exploitees():
    """Quatre départs arrêtés, c'est quatre mesures du même retard.

    N'en garder qu'une reviendrait à jeter les trois autres — et à se priver de
    la seule vraie mesure de reproductibilité disponible dans le fichier.
    """
    t = np.arange(0, 575.0, 1 / 200.0)
    rng = np.random.default_rng(11)
    ref = _quatre_departs_arretes(t) + rng.normal(0, 6.0, t.size)
    mesure = _quatre_departs_arretes(t - 0.040) + rng.normal(0, 6.0, t.size)

    recal = M.recalage_temporel(t, ref, mesure, PE, ParamsIntercorrelation())
    assert recal.non_calculable is None
    assert len(recal.fenetres) == 4, (
        f"{len(recal.fenetres)} fenêtre(s) exploitée(s) : les quatre fronts doivent "
        "contribuer"
    )
    assert recal.retard_ms == pytest.approx(40.0, abs=8.0)
    assert "fenêtres" in recal.source_dispersion
    # Chaque fenêtre couvre un départ distinct.
    debuts = sorted(f[0][0] for f in recal.fenetres)
    for attendu, obtenu in zip((10.0, 150.0, 290.0, 430.0), debuts):
        assert abs(obtenu - attendu) < 15.0


def test_le_retard_resiste_a_une_fenetre_aberrante():
    """La médiane protège ; et le désaccord entre fenêtres est annoncé."""
    t = np.arange(0, 575.0, 1 / 200.0)
    rng = np.random.default_rng(12)
    ref = _quatre_departs_arretes(t) + rng.normal(0, 6.0, t.size)
    mesure = _quatre_departs_arretes(t - 0.040) + rng.normal(0, 6.0, t.size)
    # Un seul départ mal synchronisé — un décrochage d'acquisition, par exemple.
    faux = (t >= 285.0) & (t < 330.0)
    mesure[faux] = np.interp(t[faux] - 0.300, t, _quatre_departs_arretes(t - 0.040))

    recal = M.recalage_temporel(t, ref, mesure, PE, ParamsIntercorrelation())
    assert recal.retard_ms == pytest.approx(40.0, abs=10.0), (
        "la médiane doit ignorer la fenêtre aberrante"
    )
    assert recal.faiblement_identifie, "le désaccord entre fenêtres doit être signalé"


@pytest.mark.parametrize("graine", range(6))
def test_un_retard_porte_par_le_seul_front_est_annonce_comme_fragile(graine):
    """Une seule fenêtre exploitable : on retombe sur le découpage en blocs.

    Le palier de rétro est ici parfaitement plat — il ne franchit pas le seuil
    d'activité et n'est donc pas une fenêtre. Tout repose sur le front.
    """
    plat = lambda u: _forme_depart_arrete(u, ondulation_retro=0.0)  # noqa: E731
    t, ref, mesure = _deux_voies(plat, 0.040, graine=graine)
    recal = M.recalage_temporel(t, ref, mesure, PE, ParamsIntercorrelation())
    assert len(recal.fenetres) == 1, "le palier plat ne doit pas être une fenêtre"
    assert "sous-fenêtres" in recal.source_dispersion
    assert recal.faiblement_identifie, (
        "sur un départ arrêté, seul le front porte l'information de synchronisation : "
        f"les sous-fenêtres ne peuvent pas s'accorder (étendue "
        f"{recal.dispersion_blocs_ms:.1f} ms)"
    )


@pytest.mark.parametrize("graine", range(6))
def test_un_cycle_a_variations_continues_n_est_pas_marque_fragile(graine):
    """L'indicateur ne doit pas se déclencher là où le retard est bien assis."""
    t, ref = _signal_dynamique()
    rng = np.random.default_rng(graine)
    ref = ref + rng.normal(0, 6.0, t.size)
    mesure = np.interp(t - 0.040, t, ref, left=ref[0], right=ref[-1])
    mesure = mesure + rng.normal(0, 6.0, t.size)

    recal = M.recalage_temporel(t, ref, mesure, PE, ParamsIntercorrelation())
    assert recal.retard_ms == pytest.approx(40.0, abs=3.0)
    assert not recal.faiblement_identifie, (
        f"étendue inter-blocs {recal.dispersion_blocs_ms:.1f} ms"
    )
    # Sur une fenêtre longue, la coupure demandée est respectée à l'identique.
    assert recal.coupure_passe_haut_Hz == pytest.approx(0.2)


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


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def test_la_visualisation_groupe_les_courbes_par_unite():
    """Jamais deux échelles verticales : une unité, un panneau."""
    from amdec_correlation import graphiques

    t = np.linspace(0, 10, 200)
    courbes = {
        "couple_G": 100 * np.sin(t), "couple_D": 100 * np.cos(t),
        "regime": 2000 + 500 * np.sin(t), "temperature": 20 + t,
    }
    unites = {"couple_G": "N.m", "couple_D": "N.m", "regime": "rpm", "temperature": "degC"}

    figure = graphiques.figure_visualisation(t, courbes, unites)
    # Trois unités distinctes → trois panneaux.
    assert len(figure.axes) == 3
    assert [ax.get_ylabel() for ax in figure.axes] == ["N.m", "rpm", "degC"]


def test_une_courbe_seule_est_nommee_par_le_titre_sans_legende():
    """Règle de palette : une série unique n'a pas besoin d'encadré de légende."""
    from amdec_correlation import graphiques

    t = np.linspace(0, 5, 50)
    figure = graphiques.figure_visualisation(t, {"couple": t}, {"couple": "N.m"})
    ax = figure.axes[0]
    assert ax.get_title(loc="left") == "couple"
    assert ax.get_legend() is None


class _ZonesFactices:
    """Zones minimales, en secondes, telles que la détection les publie."""

    paliers = ()
    plages_dynamiques_s = ((10.0, 20.0), (50.0, 62.0))
    plages_ecartees_s = ((70.0, 80.0),)
    plages_repos_s = ((90.0, 99.0),)


def test_les_zones_visibles_sont_rognees_a_la_fenetre_affichee():
    """Une bande à cheval sur le bord est rognée ; une bande hors champ disparaît."""
    from amdec_correlation import graphiques

    t = np.linspace(15.0, 55.0, 400)  # ne montre qu'une partie des deux plages
    bandes = graphiques.zones_visibles(t, _ZonesFactices(), None)

    familles = [cle for _, _, cle in bandes]
    assert familles == ["dynamique", "dynamique"], (
        "les plages de repos et écartées sont hors de la fenêtre : elles ne "
        "doivent ni être tracées, ni entrer en légende"
    )
    assert bandes[0][:2] == (15.0, 20.0)  # rognée à gauche
    assert bandes[1][:2] == (50.0, 55.0)  # rognée à droite


def test_les_zones_visibles_se_restreignent_aux_familles_demandees():
    from amdec_correlation import graphiques

    t = np.linspace(0.0, 100.0, 1000)
    toutes = graphiques.zones_visibles(t, _ZonesFactices(), None)
    repos = graphiques.zones_visibles(t, _ZonesFactices(), ["repos"])

    assert len(toutes) == 4
    assert [cle for _, _, cle in repos] == ["repos"]


def test_le_trace_interactif_reprend_la_composition_du_trace_fixe():
    """Mêmes panneaux, mêmes couleurs, mêmes zones : sinon les deux se contredisent.

    L'onglet propose les deux rendus sur les mêmes tableaux. S'ils divergeaient,
    le contrôle visuel ne prouverait plus rien — c'est tout son objet.
    """
    from amdec_correlation import graphiques
    from amdec_correlation import graphiques_interactifs as GI

    disponible, raison = GI.disponible()
    if not disponible:
        pytest.skip(raison)

    t = np.linspace(0.0, 100.0, 500)
    courbes = {"couple_G": 100 * np.sin(t), "couple_D": 100 * np.cos(t),
               "regime": 2000 + 500 * np.sin(t)}
    unites = {"couple_G": "N.m", "couple_D": "N.m", "regime": "rpm"}
    zones, familles = _ZonesFactices(), ["dynamique", "repos"]

    fixe = graphiques.figure_visualisation(t, courbes, unites, zones=zones,
                                           types_zones=familles)
    vif = GI.figure_visualisation(t, courbes, unites, zones=zones,
                                  types_zones=familles)

    # Deux unités → deux panneaux, dans les deux rendus.
    assert len(fixe.axes) == 2
    assert len({trace.yaxis for trace in vif.data if trace.x is not None
                and len(trace.x)}) == 2

    couleurs_fixes = {ligne.get_label(): ligne.get_color()
                      for ax in fixe.axes for ligne in ax.get_lines()}
    couleurs_vives = {trace.name: trace.line.color for trace in vif.data
                      if trace.name in courbes}
    assert couleurs_vives == couleurs_fixes, "une couleur suit un canal, pas un rendu"

    # Trois bandes visibles (deux dynamiques, une de repos), posées sur chacun
    # des deux panneaux.
    attendues = len(graphiques.zones_visibles(t, zones, familles))
    assert attendues == 3
    assert len(vif.layout.shapes) == attendues * 2


def test_le_glisser_bascule_entre_zoom_et_designation_de_plage():
    """Deux gestes distincts : agrandir l'existant, ou redemander des points."""
    from amdec_correlation import graphiques_interactifs as GI

    disponible, raison = GI.disponible()
    if not disponible:
        pytest.skip(raison)

    t = np.linspace(0.0, 10.0, 200)
    courbes, unites = {"couple": np.sin(t)}, {"couple": "N.m"}

    zoom = GI.figure_visualisation(t, courbes, unites, glisser="zoom")
    plage = GI.figure_visualisation(t, courbes, unites, glisser="select")

    assert zoom.layout.dragmode == "zoom"
    assert plage.layout.dragmode == "select"
    # Une plage de temps, jamais une bande de valeurs : la sélection verticale
    # n'aurait aucun sens pour recadrer l'axe des temps.
    assert plage.layout.selectdirection == "h"


def test_le_curseur_de_mesure_rend_un_echantillon_reel_jamais_interpole():
    """« N'invente aucune valeur » : la lecture au curseur ne fait pas exception."""
    from amdec_correlation import graphiques_interactifs as GI

    disponible, raison = GI.disponible()
    if not disponible:
        pytest.skip(raison)

    t = np.array([0.0, 0.1, 0.2, 0.3])
    courbes = {"couple": np.array([0.0, 10.0, 20.0, 30.0])}

    # Un instant demandé entre deux échantillons : on rend celui qui existe.
    instant, valeurs = GI.lire_au_curseur(t, courbes, 0.14)
    assert instant == pytest.approx(0.1)
    assert valeurs["couple"] == pytest.approx(10.0), (
        "14 N·m serait une interpolation : cette valeur n'a jamais été mesurée"
    )


def _campagne_factice():
    """Les objets de résultat nécessaires aux figures, sur un balayage simple."""
    niveaux = [0, 300, 600, 900, 600, 300, 0]
    t, ref = _escalier(niveaux)
    mesure = 1.01 * ref + 5.0
    paliers = M.detecter_paliers(t, ref, mesure, PE, ParamsPaliers())
    reg = M.regression([p.reference for p in paliers], [p.mesure for p in paliers])
    hyst = M.hysteresis(paliers, PE, ParamsPaliers())
    return t, ref, mesure, paliers, reg, hyst


def test_chaque_figure_du_rapport_a_son_equivalent_interactif():
    """Toutes les figures se manipulent, pas seulement la visualisation libre.

    Une figure interactive et son image fixe montrent la même chose : c'est la
    condition pour que l'une puisse servir à vérifier ce que l'autre affirme.
    """
    from amdec_correlation import graphiques_interactifs as GI

    disponible, raison = GI.disponible()
    if not disponible:
        pytest.skip(raison)

    t, ref, mesure, paliers, reg, hyst = _campagne_factice()

    figure = GI.figure_regression_balayage(paliers, reg, hyst, PE)
    noms = {trace.name for trace in figure.data}
    assert {"montée", "descente", "régression linéaire"} <= noms
    # L'encadré de valeurs reprend les chiffres de la régression, comme l'image fixe.
    assert any(f"{reg.a:.4f}" in (a.text or "") for a in figure.layout.annotations)

    recal = M.recalage_temporel(t, ref, mesure, PE, ParamsIntercorrelation())
    figure = GI.figure_recalage(t, ref, mesure, recal)
    assert len(figure.data) >= 4, "avant, après, et le pic d'intercorrélation"

    rep = M.Repetabilite(non_calculable="répétabilité non calculable : essai")
    figure = GI.figure_repetabilite(rep, PE)
    assert any("non calculable" in (a.text or "") for a in figure.layout.annotations), (
        "une grandeur non calculable s'affiche avec son motif, jamais en tracé vide"
    )

    figure = GI.figure_redondance(t, mesure, mesure - 2.0, PE)
    assert {trace.name for trace in figure.data} == {
        "transmission gauche", "transmission droite", "gauche − droite"
    }


def test_les_figures_interactives_gardent_la_palette_des_images_fixes():
    """Une couleur suit une entité, d'un rendu à l'autre comme d'une figure à l'autre."""
    from amdec_correlation import graphiques
    from amdec_correlation import graphiques_interactifs as GI

    disponible, raison = GI.disponible()
    if not disponible:
        pytest.skip(raison)

    _, _, _, paliers, reg, hyst = _campagne_factice()
    figure = GI.figure_regression_balayage(paliers, reg, hyst, PE)
    couleurs = {trace.name: trace.marker.color for trace in figure.data
                if trace.mode == "markers" and trace.name}
    assert couleurs["montée"] == graphiques.SERIE_1
    assert couleurs["descente"] == graphiques.SERIE_2


def test_sans_plotly_le_trace_interactif_le_dit_et_ne_trace_pas(monkeypatch):
    """La dépendance est facultative : son absence s'explique, elle ne plante pas."""
    from amdec_correlation import graphiques_interactifs as GI

    monkeypatch.setattr(GI, "go", None)
    monkeypatch.setattr(GI, "_ABSENCE", "No module named 'plotly'")

    disponible, raison = GI.disponible()
    assert disponible is False
    assert "pip install plotly" in raison, "la raison doit dire quoi faire"

    with pytest.raises(RuntimeError, match="plotly"):
        GI.figure_visualisation(np.linspace(0, 1, 10), {"c": np.zeros(10)}, {"c": ""})
