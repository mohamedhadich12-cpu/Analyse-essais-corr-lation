"""Validation de la logique de l'interface, sans lancer Streamlit.

L'enjeu de ces tests : garantir que la configuration produite par l'interface
est **exactement** celle qu'on écrirait à la main, donc que l'écran et la ligne
de commande donnent les mêmes résultats.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from amdec_correlation.config import Config  # noqa: E402
from amdec_correlation.interface import etat as E  # noqa: E402
from generer_mf4_synthetique import NOMS, generer  # noqa: E402


def _canaux(**surcharges):
    base = {
        "couple_mesure_gauche": NOMS["gauche"],
        "couple_mesure_droite": NOMS["droite"],
        "couple_reference_gauche": NOMS["reference_gauche"],
        "couple_reference_droite": NOMS["reference_droite"],
        "regime": NOMS["regime"],
        "temperature": NOMS["temperature"],
        "etat": [NOMS["etat"]],
    }
    base.update(surcharges)
    return base


def _construire(**surcharges):
    defauts = dict(
        racine="/data",
        dossier_sortie="sortie",
        pleine_echelle_Nm=1500.0,
        mode_comparaison="moyenne",
        rapport_reduction=1.0,
        incertitude_reference_k1_Nm=None,
        frequence_Hz=100.0,
        remontage_realise=False,
        canaux={"1-Balayage Couple": _canaux()},
        essais={"1-Balayage Couple": {"type": "balayage", "ligne_droite": True,
                                      "groupe_remontage": None}},
        paliers={"duree_palier_s": 2.0},
        intercorrelation={"retard_max_ms": 500.0},
        zero={"duree_fenetre_s": 5.0},
        thermique={"amplitude_min_C": 5.0, "plage_service_C": None},
        diagnostic={"seuil_gain_pc": 1.0},
    )
    defauts.update(surcharges)
    return E.construire_dict(**defauts)


# ---------------------------------------------------------------------------
# Propositions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dossier, attendu",
    [
        ("1-Balayage Couple", "balayage"),
        ("2-CPC 20°C", "repetabilite"),
        ("4-iso Pwr_traction MaV_statique", "repetabilite"),
        ("5-Décollage en pente", "dynamique"),
        ("7-WLTC", "dynamique"),
        ("3-4_1000M_DA", "dynamique"),
    ],
)
def test_type_propose_depuis_le_nom_de_dossier(dossier, attendu):
    assert E.type_propose(dossier) == attendu


def test_type_propose_retombe_sur_dynamique_si_rien_ne_correspond():
    """`dynamique` ne présuppose pas de paliers : c'est le repli le moins risqué."""
    assert E.type_propose("essai_sans_nom_parlant") == "dynamique"


def test_mapping_propose_retrouve_les_roles_depuis_les_libelles():
    mapping = E.mapping_propose(list(NOMS.values()))
    assert mapping["couple_mesure_gauche"] == NOMS["gauche"]
    assert mapping["couple_mesure_droite"] == NOMS["droite"]
    assert mapping["couple_reference_gauche"] == NOMS["reference_gauche"]
    assert mapping["couple_reference_droite"] == NOMS["reference_droite"]
    assert mapping["regime"] == NOMS["regime"]
    assert mapping["temperature"] == NOMS["temperature"]
    assert NOMS["etat"] in mapping["etat"]


def test_les_voies_de_reference_ne_sont_pas_prises_pour_des_voies_mesurees():
    """`Trq_Ref_BancGMP_G` porte « couple » et « gauche » : sans exclusion des
    mots de référence, il serait proposé comme couple mesuré gauche."""
    mapping = E.mapping_propose(list(NOMS.values()))
    assert mapping["couple_mesure_gauche"] != NOMS["reference_gauche"]
    assert mapping["couple_mesure_droite"] != NOMS["reference_droite"]
    # Et une référence latéralisée ne doit pas remplir le rôle de voie unique.
    assert mapping["couple_reference"] is None


def test_mapping_propose_laisse_vide_ce_qu_il_ne_reconnait_pas():
    mapping = E.mapping_propose(["Signal_A", "Signal_B"])
    assert all(mapping[role] is None for role in
               ("couple_mesure_gauche", "couple_reference", "regime", "temperature"))
    assert mapping["etat"] == []


# ---------------------------------------------------------------------------
# Construction de la configuration
# ---------------------------------------------------------------------------


def test_la_configuration_construite_est_acceptee_par_la_bibliotheque():
    cfg, erreurs = E.valider(_construire())
    assert erreurs == []
    assert cfg is not None
    assert cfg.pleine_echelle_Nm == 1500.0
    assert cfg.remontage_realise is False
    mapping = cfg.canaux_pour("1-Balayage Couple")
    assert mapping.voies_couple_reference == (
        NOMS["reference_gauche"], NOMS["reference_droite"]
    )
    assert mapping.a_reference is True
    assert mapping.etat == (NOMS["etat"],)


def test_le_canal_marque_absent_devient_null():
    configuration = _construire(
        canaux={"1-Balayage Couple": _canaux(temperature=E.CANAL_ABSENT)}
    )
    cfg, erreurs = E.valider(configuration)
    assert erreurs == []
    assert cfg.canaux_pour("1-Balayage Couple").temperature is None


def test_un_type_invalide_est_refuse_avec_un_message_clair():
    with pytest.raises(ValueError, match="type"):
        _construire(essais={"X": {"type": "inconnu"}})


def test_un_groupe_de_remontage_vide_devient_null():
    configuration = _construire(
        essais={"1-Balayage Couple": {"type": "balayage", "groupe_remontage": "   "}}
    )
    cfg, _ = E.valider(configuration)
    assert cfg.essais[0].groupe_remontage is None


def test_valider_renvoie_l_erreur_sans_lever():
    """L'interface doit pouvoir afficher l'erreur, jamais planter dessus."""
    incomplete = {"racine_donnees": "/x", "essais": {"A": {"type": "balayage"}}}
    cfg, erreurs = E.valider(incomplete)  # pleine_echelle_Nm manquante
    assert cfg is None
    assert erreurs and "pleine_echelle_Nm" in erreurs[0]


# ---------------------------------------------------------------------------
# Aller-retour YAML
# ---------------------------------------------------------------------------


def test_le_yaml_exporte_se_relit_a_l_identique(tmp_path):
    """La configuration exportée doit rejouer telle quelle en ligne de commande."""
    configuration = _construire(incertitude_reference_k1_Nm=2.5)
    chemin = tmp_path / "correlation.yaml"
    chemin.write_text(E.vers_yaml(configuration), encoding="utf-8")

    relu = Config.charger(chemin)
    attendu = Config.depuis_dict(configuration)
    assert relu.pleine_echelle_Nm == attendu.pleine_echelle_Nm
    assert relu.mode_comparaison == attendu.mode_comparaison
    assert relu.incertitude_reference_k1_Nm == attendu.incertitude_reference_k1_Nm
    assert relu.remontage_realise == attendu.remontage_realise
    assert [e.dossier for e in relu.essais] == [e.dossier for e in attendu.essais]
    assert relu.canaux_pour("1-Balayage Couple") == attendu.canaux_pour("1-Balayage Couple")


def test_etat_depuis_dict_rabat_le_mapping_par_defaut_sur_chaque_dossier():
    """L'interface travaille dossier par dossier : aucune valeur héritée invisible."""
    brut = {
        "canaux": {
            "defaut": {"couple_reference": "Ref", "regime": "N"},
            "par_dossier": {"A": {"regime": "N_special"}},
        },
        "essais": {"A": {"type": "balayage", "ligne_droite": True}},
    }
    etat = E.etat_depuis_dict(brut)
    assert etat["canaux"]["A"]["couple_reference"] == "Ref"   # hérité du défaut
    assert etat["canaux"]["A"]["regime"] == "N_special"       # surcharge du dossier
    assert etat["essais"]["A"]["ligne_droite"] is True


def test_etat_initial_propose_un_mapping_et_un_type_par_dossier(tmp_path):
    racine = tmp_path / "donnees"
    generer(racine)
    from amdec_correlation import inventaire as I

    inventaires = {
        inv.dossier: {"noms": inv.noms_uniques}
        for inv in I.inventorier(racine, None, 1)
    }
    depart = E.etat_initial(inventaires)
    assert set(depart["canaux"]) == set(inventaires)
    assert depart["essais"]["1-Balayage Couple"]["type"] == "balayage"
    assert depart["canaux"]["7-WLTC"]["couple_reference_gauche"] == NOMS["reference_gauche"]
    # Les libellés étant identiques partout, le mapping commun doit être complet.
    assert depart["canaux_communs"]["couple_reference_droite"] == NOMS["reference_droite"]
    assert depart["canaux_communs"]["couple_mesure_gauche"] == NOMS["gauche"]


def test_le_mapping_commun_produit_un_defaut_et_aucune_surcharge():
    """Libellés uniques sur la campagne : un seul mapping, pas un par dossier."""
    configuration = _construire(canaux_communs=_canaux(), canaux={})
    assert configuration["canaux"]["par_dossier"] == {}
    assert configuration["canaux"]["defaut"]["couple_reference_gauche"] == NOMS["reference_gauche"]

    cfg, erreurs = E.valider(configuration)
    assert erreurs == []
    # Le mapping s'applique bien à l'essai déclaré, sans avoir été redéclaré.
    assert cfg.canaux_pour("1-Balayage Couple").couple_mesure_gauche == NOMS["gauche"]


def test_l_interface_produit_les_memes_resultats_que_la_ligne_de_commande(tmp_path):
    """Garantie centrale : l'écran n'est pas un second chemin de calcul."""
    racine = tmp_path / "donnees"
    generer(racine)
    from amdec_correlation.analyse import analyser

    configuration = E.construire_dict(
        racine=str(racine),
        dossier_sortie=str(tmp_path / "sortie"),
        pleine_echelle_Nm=1500.0,
        mode_comparaison="moyenne",
        rapport_reduction=1.0,
        incertitude_reference_k1_Nm=2.0,
        frequence_Hz=200.0,
        remontage_realise=False,
        canaux={"1-Balayage Couple": _canaux()},
        essais={"1-Balayage Couple": {"type": "balayage", "ligne_droite": True}},
        paliers={}, intercorrelation={}, zero={}, diagnostic={},
        thermique={"plage_service_C": 40.0},
    )
    depuis_interface = analyser(Config.depuis_dict(configuration))

    chemin = tmp_path / "cli.yaml"
    chemin.write_text(E.vers_yaml(configuration), encoding="utf-8")
    depuis_cli = analyser(Config.charger(chemin))

    a = depuis_interface.essai("1-Balayage Couple").regression
    b = depuis_cli.essai("1-Balayage Couple").regression
    assert (a.a, a.b, a.n) == (b.a, b.b, b.n)


# ---------------------------------------------------------------------------
# Explorateur de dossiers du poste
# ---------------------------------------------------------------------------


def test_l_explorateur_dit_pourquoi_il_est_indisponible():
    """Un bouton qui échoue sans expliquer se lit comme une panne de l'outil."""
    from amdec_correlation.interface import explorateur

    ouvrable, motif = explorateur.disponible()
    assert isinstance(ouvrable, bool)
    if not ouvrable:
        assert motif, "l'indisponibilité doit toujours être motivée"
        with pytest.raises(explorateur.ExplorateurIndisponible):
            explorateur.choisir_dossier()


def test_l_explorateur_ne_renvoie_jamais_de_chemin_par_defaut(monkeypatch):
    """Annuler doit donner « rien », jamais un dossier arbitraire.

    Renvoyer le dossier de départ sur une annulation ferait analyser autre chose
    que ce que l'utilisateur a demandé, sans qu'il s'en aperçoive.
    """
    import subprocess

    from amdec_correlation.interface import explorateur

    monkeypatch.setattr(explorateur, "disponible", lambda: (True, ""))
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout="  \n", stderr=""),
    )
    assert explorateur.choisir_dossier("/tmp") is None


def test_l_explorateur_remonte_le_chemin_choisi(monkeypatch, tmp_path):
    import subprocess

    from amdec_correlation.interface import explorateur

    monkeypatch.setattr(explorateur, "disponible", lambda: (True, ""))
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(
            a[0], 0, stdout=f"{tmp_path}\n", stderr=""
        ),
    )
    assert explorateur.choisir_dossier() == str(tmp_path)


# ---------------------------------------------------------------------------
# Mémoire de la dernière utilisation
# ---------------------------------------------------------------------------


def test_la_configuration_memorisee_se_relit_a_l_identique(tmp_path):
    """Le mapping ne doit pas être à redéclarer d'un lancement à l'autre."""
    configuration = _construire(
        racine=str(tmp_path), mode_comparaison="somme",
        incertitude_reference_k1_Nm=2.5, canaux={}, canaux_communs=_canaux(),
        essais={}, ligne_droite=True,
        intercorrelation={"retard_max_ms": 500.0, "n_blocs_coherence": 6,
                          "accord_blocs_max_ms": 33.0},
    )
    memoire = tmp_path / "derniere_configuration.yaml"
    memoire.write_text(E.vers_yaml(configuration), encoding="utf-8")

    relu = E.depuis_yaml(memoire)
    scalaires = E.scalaires_depuis_dict(relu)
    assert scalaires["mode"] == "somme"
    assert scalaires["u_ref"] == 2.5 and scalaires["u_ref_connue"] is True
    assert scalaires["ligne_droite"] is True
    assert (relu["canaux"]["defaut"]["couple_mesure_gauche"]
            == _canaux()["couple_mesure_gauche"])
    # Les seuils de traitement font le voyage eux aussi.
    assert scalaires["i::n_blocs_coherence"] == 6
    assert scalaires["i::accord_blocs_max_ms"] == 33.0
    # Et la configuration relue reste directement exploitable.
    config = Config.depuis_dict(relu)
    assert config.organisation_automatique is True
    assert config.intercorrelation.accord_blocs_max_ms == 33.0


def test_une_memoire_sans_mapping_ne_doit_pas_ecraser_une_memoire_qui_en_a():
    """Un lancement avant l'inventaire ne doit pas effacer le travail de la veille."""
    vide = {"canaux": {"defaut": {r: None for r in E.ORDRE_CANAUX}, "par_dossier": {}}}
    plein = {"canaux": {"defaut": _canaux(), "par_dossier": {}}}
    assert E.mapping_renseigne(plein) is True
    assert E.mapping_renseigne(vide) is False
    # Les canaux d'état seuls ne constituent pas un mapping.
    assert E.mapping_renseigne({"canaux": {"defaut": {"etat": ["LED"]}}}) is False
