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


def test_le_format_mdf4_est_enregistre():
    assert ".mf4" in L.extensions_supportees()
    assert ".mdf" in L.extensions_supportees()
    assert "MDF4 (ASAM)" in L.formats_supportes()


def test_un_format_inconnu_est_refuse_avec_un_message_exploitable():
    with pytest.raises(L.FormatNonSupporte) as erreur:
        L.lecteur_pour("essai_du_banc.aif")
    message = str(erreur.value)
    assert ".aif" in message
    assert ".mf4" in message  # le message dit ce qui EST pris en charge


def test_l_extension_est_reconnue_quelle_que_soit_la_casse(tmp_path):
    fichier = tmp_path / "ESSAI.MF4"
    fichier.write_bytes(b"")
    assert L.lecteur_pour(fichier).nom == "MDF4 (ASAM)"


def test_lister_fichiers_ne_retient_que_les_formats_connus(tmp_path):
    balayage(tmp_path / "essai_01.mf4")
    (tmp_path / "notes.txt").write_text("ceci n'est pas une acquisition", encoding="utf-8")
    (tmp_path / "capture.aif").write_bytes(b"\x00\x01\x02")
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
