"""Validation du mode automatique : un dossier d'acquisitions, aucun type déclaré.

L'enjeu : sans aucune déclaration, l'outil doit retrouver les mêmes défauts
injectés que lorsqu'on lui décrit la campagne — et le dire quand une zone
manque, plutôt que de produire une valeur.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from amdec_correlation import rapport as R  # noqa: E402
from amdec_correlation import zones as Z  # noqa: E402
from amdec_correlation.analyse import analyser, preparer  # noqa: E402
from amdec_correlation.config import Config  # noqa: E402
from generer_mf4_synthetique import (  # noqa: E402
    GAIN, HYSTERESIS_NM, NOMS, OFFSET_NM, PLEINE_ECHELLE, RETARD_S,
    SENSIBILITE_TH_NM_PAR_C, generer,
)


def _aplatir(source: Path, cible: Path) -> Path:
    """Recopie toutes les acquisitions dans UN dossier, sans sous-dossier.

    C'est le cas d'usage visé : un dossier d'essais, sans distinction de type.
    """
    cible.mkdir(parents=True, exist_ok=True)
    for fichier in sorted(source.rglob("*.mf4")):
        shutil.copy(fichier, cible / f"{fichier.parent.name}__{fichier.name}")
    return cible


def _config(racine: Path, sortie: Path, **surcharges) -> Config:
    base = {
        "racine_donnees": str(racine),
        "dossier_sortie": str(sortie),
        "pleine_echelle_Nm": PLEINE_ECHELLE,
        "comparaison": {"mode": "moyenne", "incertitude_reference_k1_Nm": 2.0},
        "acquisition": {"frequence_reechantillonnage_Hz": 200},
        "ligne_droite": True,
        "remontage": {"realise": False},
        "canaux": {"defaut": {
            "couple_mesure_gauche": NOMS["gauche"],
            "couple_mesure_droite": NOMS["droite"],
            "couple_reference_gauche": NOMS["reference_gauche"],
            "couple_reference_droite": NOMS["reference_droite"],
            "regime": NOMS["regime"],
            "temperature": NOMS["temperature"],
        }},
        "thermique": {"plage_service_C": 40.0},
        # AUCUN essai déclaré : c'est ce qui déclenche le mode automatique.
    }
    base.update(surcharges)
    return Config.depuis_dict(base)


@pytest.fixture(scope="module")
def campagne(tmp_path_factory):
    source = tmp_path_factory.mktemp("source")
    generer(source)
    racine = _aplatir(source, tmp_path_factory.mktemp("plat") / "essais")
    sortie = tmp_path_factory.mktemp("sortie")
    cfg = _config(racine, sortie)
    return analyser(cfg), cfg


# ---------------------------------------------------------------------------
# Aiguillage
# ---------------------------------------------------------------------------


def test_sans_essai_declare_le_mode_est_automatique(campagne):
    resultats, cfg = campagne
    assert cfg.organisation_automatique is True
    assert resultats.mode == "automatique"


def test_toutes_les_acquisitions_sont_lues_sans_sous_dossier(campagne):
    resultats, _ = campagne
    # 8 fichiers synthétiques, tous dans un seul dossier.
    assert len(resultats.zones) == 8
    assert all(z.erreur is None for z in resultats.zones)


# ---------------------------------------------------------------------------
# Détection des zones
# ---------------------------------------------------------------------------


def test_les_zones_sont_reconnues_pour_ce_qu_elles_sont(campagne):
    resultats, _ = campagne
    par_nom = {z.chemin.name: z for z in resultats.zones}

    balayage = next(z for n, z in par_nom.items() if "balayage" in n)
    assert balayage.alimente_regression
    assert balayage.alimente_hysteresis  # montée ET descente détectées

    wltc = next(z for n, z in par_nom.items() if "wltc" in n)
    assert wltc.alimente_retard
    assert wltc.alimente_derive_zero  # zéros de début et de fin
    assert not wltc.alimente_hysteresis

    cpc = next(z for n, z in par_nom.items() if "cpc_rep01" in n)
    assert cpc.paliers and not cpc.alimente_retard


def test_le_profil_est_lisible_pour_verification(campagne):
    resultats, _ = campagne
    balayage = next(z for z in resultats.zones if "balayage" in z.chemin.name)
    profil = balayage.profil()
    assert "paliers" in profil and "↑" in profil and "↓" in profil


def test_la_synthese_compte_les_contributeurs(campagne):
    resultats, _ = campagne
    synthese = Z.synthese(resultats.zones)
    assert synthese["fichiers"] == 8
    assert synthese["exploités"] == 8
    assert synthese["retard"] >= 2          # décollage en pente + WLTC
    assert synthese["hystérésis"] >= 1      # le balayage
    assert synthese["paliers au total"] > 20


# ---------------------------------------------------------------------------
# Les grandeurs sont retrouvées, sans déclaration
# ---------------------------------------------------------------------------


def test_le_gain_et_l_offset_sont_retrouves(campagne):
    resultats, _ = campagne
    reg = resultats.regression_globale
    assert reg is not None and reg.non_calculable is None
    assert reg.a == pytest.approx(GAIN, abs=0.003)
    # L'offset apparent inclut la demi-hystérésis du balayage, dilué par les
    # paliers des autres acquisitions.
    assert reg.b == pytest.approx(OFFSET_NM, abs=2.0)


def test_l_hysteresis_est_retrouvee(campagne):
    resultats, _ = campagne
    hyst = resultats.hysteresis_globale
    assert hyst.non_calculable is None
    assert hyst.max_pc_pe == pytest.approx(100 * HYSTERESIS_NM / PLEINE_ECHELLE, abs=0.06)


def test_le_retard_est_retrouve(campagne):
    resultats, _ = campagne
    retards = resultats.retards_ms
    assert len(retards) >= 2
    for retard in retards:
        assert retard == pytest.approx(RETARD_S * 1000.0, abs=5.0)


def test_la_sensibilite_thermique_est_retrouvee(campagne):
    resultats, _ = campagne
    th = resultats.thermique_globale
    assert th.non_calculable is None
    assert th.pente_Nm_par_C == pytest.approx(SENSIBILITE_TH_NM_PAR_C, rel=0.2)


def test_la_repetabilite_vient_des_niveaux_partages_entre_fichiers(campagne):
    """Sans déclaration, la répétition se reconnaît au niveau de couple atteint."""
    resultats, _ = campagne
    rep = resultats.repetabilite_globale
    assert rep.non_calculable is None
    assert rep.degres_liberte > 0
    assert rep.ecart_type_pc_pe > 0


def test_la_derive_de_zero_est_trouvee_sur_les_cycles(campagne):
    resultats, _ = campagne
    assert resultats.derives_zero_globales
    noms = {nom for nom, _ in resultats.derives_zero_globales}
    assert any("wltc" in n or "pente" in n for n in noms)


def test_le_bilan_d_incertitude_est_complet(campagne):
    resultats, _ = campagne
    inc = resultats.incertitude
    assert inc.non_calculable is None
    assert not inc.exclusions, f"exclusions inattendues : {inc.exclusions}"


def test_la_redondance_utilise_le_drapeau_global(campagne):
    resultats, _ = campagne
    assert resultats.redondance is not None
    assert resultats.redondance.non_calculable is None


# ---------------------------------------------------------------------------
# Restitutions et refus
# ---------------------------------------------------------------------------


def test_le_rapport_est_complet_en_mode_automatique(campagne):
    resultats, _ = campagne
    texte = R.rediger(resultats)
    for ligne in (
        "Offset b (% PE)", "Erreur de sensibilité a−1 (%)", "Non-linéarité (% PE)",
        "Hystérésis (% PE)", "Répétabilité après remontage (% PE)",
        "Retard temporel (ms)", "Sensibilité thermique (% PE pour 10 °C)",
        "Dérive de zéro sur cycle (% PE)", "Incertitude élargie résultante (k=2)",
    ):
        assert ligne in texte
    assert "non calculable" in texte  # la répétabilité après remontage


def test_le_detail_remonte_chaque_acquisition_a_sa_source(campagne):
    """Sans essai déclaré, le § détail doit rester traçable fichier par fichier."""
    resultats, _ = campagne
    texte = R.rediger(resultats)
    assert "## 10.5 Détail par acquisition" in texte
    for zone in resultats.zones:
        assert f"`{zone.chemin.name}`" in texte
    # Et les grandeurs par fichier y figurent, pas seulement la synthèse.
    assert "Recalage `" in texte and "Dérive de zéro `" in texte


def test_une_figure_de_zones_est_produite_par_acquisition(campagne):
    """Le tableau dit combien, la figure dit où : les deux doivent exister."""
    resultats, _ = campagne
    figures = {f.name for f in resultats.figures}
    zones_tracees = {n for n in figures if n.startswith("zones_")}
    assert len(zones_tracees) == len(resultats.zones)
    for figure in resultats.figures:
        assert figure.exists() and figure.stat().st_size > 5000


def test_un_dossier_sans_acquisition_est_signale(tmp_path):
    cfg = _config(tmp_path / "vide", tmp_path / "sortie")
    (tmp_path / "vide").mkdir()
    resultats = analyser(cfg)
    assert any("Aucune acquisition" in a for a in resultats.avertissements)


def test_un_mapping_incomplet_est_signale_sans_planter(tmp_path):
    source = tmp_path / "src"
    generer(source)
    racine = _aplatir(source, tmp_path / "plat")
    cfg = _config(racine, tmp_path / "sortie")
    cfg._canaux_defaut["couple_reference_gauche"] = None
    cfg._canaux_defaut["couple_reference_droite"] = None

    resultats = analyser(cfg)
    assert any("Mapping incomplet" in a for a in resultats.avertissements)


def test_des_acquisitions_sans_zone_exploitable_donnent_des_refus(tmp_path):
    """Un dossier ne contenant que des paliers : aucun retard identifiable."""
    from generer_mf4_synthetique import cpc

    racine = tmp_path / "paliers"
    racine.mkdir(parents=True)
    cpc(racine / "a.mf4", graine=1, biais_Nm=0.0)
    cpc(racine / "b.mf4", graine=2, biais_Nm=1.0)

    resultats = analyser(_config(racine, tmp_path / "sortie"))
    assert resultats.retards_ms == []
    texte = R.rediger(resultats)
    assert "Retard temporel (ms) | **non calculable**" in texte
    # Mais la régression, elle, doit être disponible.
    assert resultats.regression_globale.non_calculable is None


# ---------------------------------------------------------------------------
# Choix des zones affichées
# ---------------------------------------------------------------------------


def test_les_familles_proposees_sont_celles_reellement_presentes(campagne):
    """Proposer une famille absente ferait douter du réglage, pas du fichier."""
    resultats, _ = campagne
    par_nom = {z.chemin.name: z for z in resultats.zones}

    balayage = next(z for n, z in par_nom.items() if "balayage" in n)
    assert "montee" in balayage.familles() and "descente" in balayage.familles()
    assert "repos" not in balayage.familles()

    wltc = next(z for n, z in par_nom.items() if "wltc" in n)
    assert "dynamique" in wltc.familles() and "repos" in wltc.familles()
    assert "montee" not in wltc.familles()

    # Toutes les clés annoncées doivent exister côté figures.
    from amdec_correlation import graphiques

    for zone in resultats.zones:
        assert set(zone.familles()) <= set(graphiques.TYPES_ZONES)


def test_le_filtre_ne_trace_que_les_familles_demandees(campagne):
    """Le tracé doit obéir au filtre, sinon le choix est décoratif."""
    import matplotlib.pyplot as plt

    from amdec_correlation import graphiques

    resultats, cfg = campagne
    zone = next(z for z in resultats.zones if "balayage" in z.chemin.name)
    t = np.linspace(0, zone.duree_s, 4000)
    y = np.zeros_like(t)

    figure = graphiques.figure_zones(t, y, y, zone, cfg.pleine_echelle_Nm,
                                     types=["montee"])
    libelles = [texte.get_text() for texte in figure.axes[0].get_legend().get_texts()]
    plt.close(figure)
    assert graphiques.TYPES_ZONES["montee"] in libelles
    assert graphiques.TYPES_ZONES["descente"] not in libelles

    # Et aucune famille demandée mais absente ne s'invente une légende.
    figure = graphiques.figure_zones(t, y, y, zone, cfg.pleine_echelle_Nm,
                                     types=["repos"])
    libelles = [texte.get_text() for texte in figure.axes[0].get_legend().get_texts()]
    plt.close(figure)
    assert graphiques.TYPES_ZONES["repos"] not in libelles
