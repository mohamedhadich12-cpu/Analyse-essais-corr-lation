"""Aiguillage des formats d'acquisition.

Le pipeline ne connaît qu'une chose des fichiers d'essai : un **lecteur** sait
décrire leurs canaux et charger des signaux sur une base de temps commune. Le
format concret — MDF4 aujourd'hui, d'autres ensuite — est isolé derrière cette
interface.

Ajouter un format revient à écrire deux fonctions et à les enregistrer :

    from .lecteurs import Lecteur, enregistrer

    enregistrer(Lecteur(
        nom="Mon format",
        extensions=(".ext",),
        decrire=ma_fonction_decrire,     # (chemin) -> list[DescriptionCanal]
        charger=ma_fonction_charger,     # (chemin, mapping, frequence) -> SignauxEssai
    ))

Rien d'autre ne change : inventaire, mapping, traitements et rapport
fonctionnent à l'identique quel que soit le format d'origine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

__all__ = [
    "Lecteur",
    "enregistrer",
    "lecteur_pour",
    "extensions_supportees",
    "formats_supportes",
    "lister_fichiers",
    "FormatNonSupporte",
]


class FormatNonSupporte(RuntimeError):
    """Extension sans lecteur enregistré. Le message liste les formats connus."""


@dataclass(frozen=True)
class Lecteur:
    """Un format d'acquisition et les deux fonctions qui permettent de l'exploiter."""

    nom: str
    extensions: tuple[str, ...]
    decrire: Callable[..., list]
    charger: Callable[..., object]

    def __post_init__(self) -> None:
        if not self.extensions:
            raise ValueError(f"Lecteur « {self.nom} » : au moins une extension est requise.")


_LECTEURS: dict[str, Lecteur] = {}


def enregistrer(lecteur: Lecteur) -> None:
    """Déclare un lecteur pour ses extensions (comparaison insensible à la casse)."""
    for extension in lecteur.extensions:
        _LECTEURS[extension.lower()] = lecteur


def lecteur_pour(chemin: str | Path) -> Lecteur:
    """Lecteur associé à l'extension du fichier."""
    extension = Path(chemin).suffix.lower()
    lecteur = _LECTEURS.get(extension)
    if lecteur is None:
        connus = ", ".join(sorted(extensions_supportees())) or "aucun"
        raise FormatNonSupporte(
            f"Aucun lecteur pour l'extension « {extension or '(sans extension)'} » "
            f"({Path(chemin).name}). Formats pris en charge : {connus}."
        )
    return lecteur


def extensions_supportees() -> set[str]:
    return set(_LECTEURS)


def formats_supportes() -> dict[str, tuple[str, ...]]:
    """Nom de format -> extensions, pour l'affichage à l'utilisateur."""
    par_nom: dict[str, tuple[str, ...]] = {}
    for lecteur in set(_LECTEURS.values()):
        par_nom[lecteur.nom] = lecteur.extensions
    return par_nom


def lister_fichiers(dossier: str | Path, max_fichiers: int | None = None) -> list[Path]:
    """Acquisitions exploitables d'un sous-dossier d'essai, tous formats confondus.

    La recherche est récursive et insensible à la casse de l'extension.
    """
    dossier = Path(dossier)
    if not dossier.is_dir():
        return []
    fichiers = sorted(
        {
            chemin
            for chemin in dossier.rglob("*")
            if chemin.is_file() and chemin.suffix.lower() in _LECTEURS
        },
        key=lambda p: p.name.lower(),
    )
    return fichiers[:max_fichiers] if max_fichiers else fichiers


# -- fonctions d'accès, aiguillées vers le lecteur du format -----------------


def decrire_canaux(chemin: str | Path) -> list:
    """Inventorie les canaux d'un fichier, quel que soit son format."""
    return lecteur_pour(chemin).decrire(chemin)


def charger_signaux(chemin: str | Path, mapping, frequence_Hz: float):
    """Charge les canaux mappés sur une grille de temps commune."""
    return lecteur_pour(chemin).charger(chemin, mapping, frequence_Hz)


def _enregistrer_lecteurs_integres() -> None:
    """Enregistre les lecteurs livrés avec le paquet.

    L'import est fait ici, et non au sommet du module, pour éviter une
    dépendance circulaire : le lecteur MDF4 importe ce module pour s'y déclarer.
    """
    from . import io_mdf  # noqa: F401  (l'import provoque l'enregistrement)


_enregistrer_lecteurs_integres()
