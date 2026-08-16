"""L'acquisition « toutes zones » exerce-t-elle bien toutes les situations ?

Ce fichier synthétique sert à essayer l'application à la main : il n'a d'intérêt
que s'il porte réellement chaque famille de zones et chaque défaut annoncé par
sa fiche de vérité terrain. Les tests ci-dessous le vérifient — sans quoi la
fiche promettrait des cas que le fichier ne contient pas, ce qui est pire que
pas de fichier du tout.

Ils valident aussi les deux comportements que ce fichier doit démontrer :
  * le verrou de durée minimale des fenêtres d'intercorrélation, et sa levée ;
  * l'arbitrage entre plages retenues et plages écartées.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from amdec_correlation import graphiques  # noqa: E402
from amdec_correlation import zones as Z  # noqa: E402
from amdec_correlation.analyse import analyser, preparer  # noqa: E402
from amdec_correlation.config import Config  # noqa: E402
from generer_mf4_synthetique import (  # noqa: E402
    ECART_VOIES_NM,
    GAIN,
    HYSTERESIS_NM,
    NOMS,
    PLEINE_ECHELLE,
    RETARD_S,
    SENSIBILITE_TH_NM_PAR_C,
)
from generer_mf4_toutes_zones import (  # noqa: E402
    TEMPERATURE_CHAUDE,
    TEMPERATURE_FROIDE,
    generer,
)


def _config(racine: Path, sortie: Path, **intercorrelation) -> Config:
    return Config.depuis_dict({
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
            "etat": [NOMS["etat"]],
        }},
        "thermique": {"plage_service_C": 40.0},
        "intercorrelation": intercorrelation or {},
    })


@pytest.fixture(scope="module")
def dossier(tmp_path_factory) -> Path:
    racine = tmp_path_factory.mktemp("toutes_zones")
    generer(racine)
    return racine


@pytest.fixture(scope="module")
def campagne(dossier, tmp_path_factory):
    cfg = _config(dossier, tmp_path_factory.mktemp("sortie"))
    return analyser(cfg), cfg


def _zones(dossier: Path, cfg: Config):
    fichier = next(dossier.glob("*.mf4"))
    return Z.detecter(preparer(fichier, cfg, cfg.canaux_pour("")), cfg)


# ---------------------------------------------------------------------------
# Couverture des familles de zones
# ---------------------------------------------------------------------------


def test_le_fichier_porte_les_cinq_familles_de_zones(campagne):
    """Une seule acquisition, toutes les familles : c'est sa raison d'être."""
    resultats, _ = campagne
    zone = resultats.zones[0]
    assert set(zone.familles()) == {
        "montee", "descente", "dynamique", "ecartee", "repos"
    }, zone.profil()
    # Toutes les familles annoncées sont traçables : une famille détectée que la
    # figure ne saurait pas dessiner ne serait vérifiable nulle part.
    assert set(zone.familles()) <= set(graphiques.TYPES_ZONES)


def test_les_deux_sens_de_balayage_sont_presents(campagne):
    """Sans les deux sens, l'hystérésis ne serait pas calculable."""
    resultats, _ = campagne
    paliers = resultats.zones[0].paliers
    assert sum(p.sens == "montee" for p in paliers) >= 8
    assert sum(p.sens == "descente" for p in paliers) >= 8


def test_le_retro_des_departs_arretes_ne_passe_pas_pour_un_palier(campagne):
    """Les longues sections plates à −150 N·m doivent rester hors régression.

    Elles sont agitées exprès. Si l'agitation venait à disparaître, elles
    entreraient dans la régression comme un point de mesure à couple négatif
    relevé à chaud, et le gain s'écarterait sans que rien ne soit en cause.
    """
    resultats, _ = campagne
    niveaux = {round(p.reference) for p in resultats.zones[0].paliers}
    assert not any(niveau < -20 for niveau in niveaux), niveaux


def test_les_trois_plages_de_repos_sont_vues_dont_celle_du_milieu(campagne):
    """Début, milieu, fin : seules les deux extrêmes ferment la dérive de zéro."""
    resultats, cfg = campagne
    repos = resultats.zones[0].plages_repos_s
    duree = resultats.zones[0].duree_s
    assert len(repos) == 3, repos
    assert repos[0][0] < 0.25 * duree, "il faut un relevé de zéro en début d'essai"
    assert repos[-1][1] > 0.75 * duree, "il faut un relevé de zéro en fin d'essai"
    assert 0.25 * duree < repos[1][0] < 0.75 * duree, (
        "la plage du milieu doit exister : c'est elle qui montre que la position "
        "compte, et pas seulement la présence"
    )


# ---------------------------------------------------------------------------
# Ce que le fichier doit démontrer
# ---------------------------------------------------------------------------


def test_le_verrou_de_duree_minimale_se_constate_puis_se_leve(dossier, tmp_path):
    """Les départs arrêtés n'apparaissent qu'une fois la durée minimale abaissée.

    C'est le principal piège de réglage sur une campagne de départs arrêtés :
    ce fichier doit permettre de le voir, et de vérifier que le levier marche.
    """
    cfg_defaut = _config(dossier, tmp_path / "a")
    cfg_abaisse = _config(dossier, tmp_path / "b", duree_min_fenetre_s=2.0)

    par_defaut = _zones(dossier, cfg_defaut).plages_dynamiques_s
    abaisse = _zones(dossier, cfg_abaisse).plages_dynamiques_s

    assert len(par_defaut) == 1, (
        "avec le réglage par défaut, seul le cycle transitoire est assez long"
    )
    assert len(abaisse) == 1 + 4, (
        "à 2 s, les quatre départs arrêtés rejoignent le cycle : "
        f"{[(round(a), round(b)) for a, b in abaisse]}"
    )


def test_une_plage_active_mais_pauvre_est_ecartee(campagne):
    """Franchir le seuil d'activité ne suffit pas : encore faut-il apprendre."""
    resultats, _ = campagne
    zone = resultats.zones[0]
    assert len(zone.plages_ecartees_s) == 1, zone.profil()
    debut, fin = zone.plages_ecartees_s[0]
    retenues = zone.plages_dynamiques_s
    assert all(not (r[0] <= debut < r[1]) for r in retenues), (
        "la plage écartée ne doit pas chevaucher une plage retenue"
    )


# ---------------------------------------------------------------------------
# Vérité terrain
# ---------------------------------------------------------------------------


def test_les_defauts_injectes_sont_retrouves(campagne):
    """Les grandeurs annoncées par la fiche doivent sortir du pipeline."""
    resultats, _ = campagne

    regression = resultats.regression_globale
    assert regression is not None and regression.non_calculable is None
    # La pente est tirée vers le bas par le zéro final, relevé à chaud : on
    # vérifie l'ordre de grandeur, pas l'égalité — la fiche l'explique.
    assert regression.a == pytest.approx(GAIN, abs=0.002)
    assert regression.r2 > 0.999

    hysteresis = resultats.hysteresis_globale
    assert hysteresis.non_calculable is None
    assert hysteresis.moyenne_Nm == pytest.approx(HYSTERESIS_NM, abs=0.6)

    assert len(resultats.recalages_globaux) == 1
    _, recalage = resultats.recalages_globaux[0]
    assert recalage.non_calculable is None
    assert recalage.retard_ms == pytest.approx(1000 * RETARD_S, abs=3.0)

    _, derive = resultats.derives_zero_globales[0]
    # La dérive du zéro est entièrement d'origine thermique : le rotor s'échauffe
    # entre le relevé de zéro initial et le relevé final.
    attendu_Nm = SENSIBILITE_TH_NM_PAR_C * (TEMPERATURE_CHAUDE - TEMPERATURE_FROIDE)
    assert derive.derive_Nm == pytest.approx(attendu_Nm, abs=0.5)

    thermique = resultats.thermique_globale
    assert thermique.non_calculable is None
    assert thermique.pente_Nm_par_C == pytest.approx(SENSIBILITE_TH_NM_PAR_C, abs=0.05)

    redondance = resultats.redondance
    assert redondance.non_calculable is None
    assert redondance.moyenne_Nm == pytest.approx(ECART_VOIES_NM, abs=0.2)


def test_tout_est_en_newton_metres_et_rien_n_est_converti_deux_fois(campagne):
    """Aucune grandeur ne traîne une conversion en pourcentage de pleine échelle.

    Le piège est silencieux : une valeur déjà en N·m à laquelle on applique
    encore la conversion depuis les % PE sort quinze fois trop grande sur un
    capteur de 1500 N·m, sans que rien ne plante. Le bilan d'incertitude est le
    meilleur témoin — il combine cinq grandeurs, et une seule mal convertie le
    fait sortir de l'ordre de grandeur du capteur.
    """
    resultats, cfg = campagne

    inc = resultats.incertitude
    assert inc.non_calculable is None
    for contribution in inc.contributions:
        assert abs(contribution.valeur_Nm) < 50.0, (
            f"« {contribution.nom} » vaut {contribution.valeur_Nm:.1f} N·m : "
            "c'est l'ordre de grandeur d'une valeur convertie deux fois"
        )
    # Une chaîne à 1 % d'erreur sur 1500 N·m ne peut pas porter une incertitude
    # élargie de plusieurs dizaines de N·m.
    assert inc.U_k2_Nm < 20.0

    # Les grandeurs élémentaires sont bien du même ordre que ce qui est injecté.
    assert 1.0 < resultats.hysteresis_globale.max_Nm < 10.0
    _, derive = resultats.derives_zero_globales[0]
    assert 1.0 < abs(derive.derive_Nm) < 10.0
    assert 0.0 < resultats.non_linearite_Nm < 20.0


def test_une_ancienne_configuration_en_pourcentage_est_reprise(tmp_path, dossier):
    """Un réglage enregistré en % PE doit continuer de se relire, et le dire.

    Refuser la clé aurait fait perdre le réglage d'une campagne ; la convertir
    en silence aurait changé ce que la détection voit sans que personne ne le
    sache. On convertit, et on l'annonce.
    """
    ancienne = {
        "racine_donnees": str(dossier),
        "pleine_echelle_Nm": PLEINE_ECHELLE,
        "paliers": {"tolerance_stab_pc_pe": 0.5},
        "intercorrelation": {"seuil_activite_pc_pe": 2.0},
    }
    cfg = Config.depuis_dict(ancienne)

    assert cfg.paliers.tolerance_stab_Nm == pytest.approx(7.5)
    assert cfg.intercorrelation.seuil_activite_Nm == pytest.approx(30.0)
    assert len(cfg.seuils_repris) == 2
    for avis in cfg.seuils_repris:
        assert "ancien réglage" in avis and "N·m" in avis


def test_les_grandeurs_multi_acquisitions_sont_declarees_non_calculables(campagne):
    """Un fichier unique ne peut pas porter une répétabilité : il faut le dire.

    C'est le comportement que ce fichier doit démontrer — refuser plutôt
    qu'estimer — et pas une lacune du générateur.
    """
    resultats, _ = campagne
    assert resultats.repetabilite_globale.non_calculable
    assert "acquisitions différentes" in resultats.repetabilite_globale.non_calculable
    assert resultats.motif_remontage


def test_la_fiche_de_verite_accompagne_le_fichier(dossier):
    """Sans la fiche, les valeurs à retrouver ne sont écrites nulle part."""
    fiche = dossier / "verite_terrain.md"
    assert fiche.is_file()
    texte = fiche.read_text(encoding="utf-8")
    assert "Données fabriquées" in texte
    for attendu in ("gain", "Hystérésis", "Retard", "Dérive de zéro"):
        assert attendu in texte
