"""Validation de l'aiguillage des formats d'acquisition.

Ces tests garantissent qu'ajouter un format ne demande que deux fonctions, et
qu'un format inconnu est refusé avec un message exploitable plutôt qu'ignoré en
silence — un fichier d'essai qui disparaît sans bruit de l'inventaire serait la
pire des défaillances.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from amdec_correlation import lecteurs as L  # noqa: E402
from generer_mf4_synthetique import balayage  # noqa: E402


def test_les_extensions_de_conteneur_mdf_sont_enregistrees():
    for extension in (".mf4", ".mdf", ".aif"):
        assert extension in L.extensions_supportees()


def test_un_format_inconnu_est_refuse_avec_un_message_exploitable():
    with pytest.raises(L.FormatNonSupporte) as erreur:
        L.lecteur_pour("essai_du_banc.xyz")
    message = str(erreur.value)
    assert ".xyz" in message
    assert ".mf4" in message  # le message dit ce qui EST pris en charge


# ---------------------------------------------------------------------------
# Exports ETAS INCA (.aif)
# ---------------------------------------------------------------------------


def test_un_aif_conteneur_mdf_est_lu_comme_un_mf4(tmp_path):
    """Les exports INCA sont des conteneurs MDF sous une autre extension."""
    from amdec_correlation.config import MappingCanaux
    from generer_mf4_synthetique import NOMS

    source = tmp_path / "essai.mf4"
    balayage(source)
    aif = tmp_path / "essai_inca.aif"
    aif.write_bytes(source.read_bytes())

    canaux = {d.nom for d in L.decrire_canaux(aif)}
    assert NOMS["gauche"] in canaux and NOMS["reference_gauche"] in canaux

    mapping = MappingCanaux(
        couple_mesure_gauche=NOMS["gauche"],
        couple_reference_gauche=NOMS["reference_gauche"],
    )
    signaux = L.charger_signaux(aif, mapping, 100.0)
    assert signaux.a("couple_mesure_gauche") is not None


def test_les_aif_sont_pris_dans_l_inventaire(tmp_path):
    source = tmp_path / "essai.mf4"
    balayage(source)
    (tmp_path / "autre.aif").write_bytes(source.read_bytes())
    assert {f.name for f in L.lister_fichiers(tmp_path)} == {"essai.mf4", "autre.aif"}


def test_un_aif_qui_n_est_pas_du_mdf_est_refuse_explicitement(tmp_path):
    """Le contrôle porte sur le CONTENU : une extension ne prouve rien.

    Refuser franchement vaut mieux que produire des signaux plausibles et faux —
    une erreur de ce genre traverserait un rapport AMDEC sans être vue.
    """
    from amdec_correlation.io_mdf import ErreurChargement

    faux = tmp_path / "capture.aif"
    faux.write_bytes(b"FORM\x00\x00\x10\x00AIFF" + b"\x00" * 64)  # en-tête audio AIFF

    with pytest.raises(ErreurChargement) as erreur:
        L.decrire_canaux(faux)
    message = str(erreur.value)
    assert "n'est pas un conteneur MDF" in message
    assert "46 4f 52 4d" in message  # les premiers octets, pour identification


@pytest.mark.parametrize("nom", ["ESSAI.MF4", "essai.mf4", "Essai.Mdf", "RELEVE.AIF"])
def test_l_extension_est_reconnue_quelle_que_soit_la_casse(tmp_path, nom):
    fichier = tmp_path / nom
    fichier.write_bytes(b"")
    assert L.lecteur_pour(fichier).extensions == (".mf4", ".mdf", ".aif")


def test_lister_fichiers_ne_retient_que_les_formats_connus(tmp_path):
    balayage(tmp_path / "essai_01.mf4")
    (tmp_path / "notes.txt").write_text("ceci n'est pas une acquisition", encoding="utf-8")
    (tmp_path / "config.xml").write_text("<x/>", encoding="utf-8")
    (tmp_path / "ESSAI_02.MF4").write_bytes((tmp_path / "essai_01.mf4").read_bytes())

    trouves = {f.name for f in L.lister_fichiers(tmp_path)}
    assert trouves == {"essai_01.mf4", "ESSAI_02.MF4"}


def test_lister_fichiers_est_recursif(tmp_path):
    balayage(tmp_path / "session_a" / "essai.mf4")
    assert len(L.lister_fichiers(tmp_path)) == 1


def test_un_lecteur_ajoute_est_immediatement_pris_en_compte(tmp_path):
    """Le contrat d'extension : deux fonctions et un enregistrement suffisent."""
    appels: list[Path] = []

    def decrire_factice(chemin):
        appels.append(Path(chemin))
        return []

    lecteur = L.Lecteur(
        nom="Format de démonstration", extensions=(".demo",),
        decrire=decrire_factice, charger=lambda *a, **k: None,
    )
    anciennes = dict(L._LECTEURS)
    try:
        L.enregistrer(lecteur)
        fichier = tmp_path / "essai.demo"
        fichier.write_bytes(b"")

        assert L.lecteur_pour(fichier).nom == "Format de démonstration"
        assert [f.name for f in L.lister_fichiers(tmp_path)] == ["essai.demo"]
        assert L.decrire_canaux(fichier) == []
        assert appels == [fichier]
    finally:
        L._LECTEURS.clear()
        L._LECTEURS.update(anciennes)


def test_un_lecteur_sans_extension_est_refuse():
    with pytest.raises(ValueError, match="extension"):
        L.Lecteur(nom="Vide", extensions=(), decrire=lambda p: [], charger=lambda *a: None)


def test_le_chargement_est_aiguille_vers_le_bon_lecteur(tmp_path):
    """L'inventaire passe par l'aiguillage, pas directement par le lecteur MDF."""
    from amdec_correlation.config import MappingCanaux
    from generer_mf4_synthetique import NOMS

    fichier = tmp_path / "essai.mf4"
    balayage(fichier)
    mapping = MappingCanaux(
        couple_mesure_gauche=NOMS["gauche"],
        couple_reference_gauche=NOMS["reference_gauche"],
    )
    signaux = L.charger_signaux(fichier, mapping, 100.0)
    assert signaux.a("couple_mesure_gauche") is not None
    assert signaux.t.size > 0


def test_un_essai_present_dans_deux_formats_est_signale(tmp_path):
    """Le doublon de format ne fait échouer aucun calcul : il les fausse.

    Sans avertissement, le même essai serait compté deux fois dans tout ce qui
    se cumule, sans qu'aucune erreur ne le révèle.
    """
    source = tmp_path / "releve_01.mf4"
    balayage(source)
    (tmp_path / "releve_01.aif").write_bytes(source.read_bytes())
    (tmp_path / "releve_02.mf4").write_bytes(source.read_bytes())

    doublons = L.doublons_de_format(L.lister_fichiers(tmp_path))
    assert set(doublons) == {"releve_01"}
    assert {f.suffix for f in doublons["releve_01"]} == {".mf4", ".aif"}


def test_le_doublon_remonte_dans_les_avertissements_de_la_campagne(tmp_path):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "src"))
    from amdec_correlation.analyse import analyser
    from amdec_correlation.config import Config
    from generer_mf4_synthetique import NOMS

    racine = tmp_path / "donnees" / "1-Balayage Couple"
    racine.mkdir(parents=True)
    balayage(racine / "essai.mf4")
    (racine / "essai.aif").write_bytes((racine / "essai.mf4").read_bytes())

    cfg = Config.depuis_dict({
        "racine_donnees": str(tmp_path / "donnees"),
        "dossier_sortie": str(tmp_path / "sortie"),
        "pleine_echelle_Nm": 1500.0,
        "acquisition": {"frequence_reechantillonnage_Hz": 200},
        "canaux": {"defaut": {
            "couple_mesure_gauche": NOMS["gauche"],
            "couple_mesure_droite": NOMS["droite"],
            "couple_reference_gauche": NOMS["reference_gauche"],
            "couple_reference_droite": NOMS["reference_droite"],
        }},
        "essais": {"1-Balayage Couple": {"type": "balayage"}},
    })
    campagne = analyser(cfg)
    assert any("sous 2 formats" in a for a in campagne.avertissements)
