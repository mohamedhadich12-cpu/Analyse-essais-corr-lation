"""État de l'interface : assemblage de la configuration, sans dépendance à Streamlit.

Isoler cette logique de l'affichage la rend testable sans lancer d'interface, et
garantit que l'interface produit exactement la même configuration YAML que celle
qu'on écrirait à la main — donc exactement les mêmes résultats que la ligne de
commande.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import yaml

from ..config import CANAUX_SCALAIRES, TYPES_ESSAI, Config
from ..inventaire import candidats

# Valeur affichée dans les listes déroulantes pour un canal volontairement absent.
CANAL_ABSENT = "— absent —"

LIBELLES_CANAUX = {
    "couple_mesure_gauche": "Couple mesuré — transmission gauche",
    "couple_mesure_droite": "Couple mesuré — transmission droite",
    "couple_reference_gauche": "Couple de référence banc GMP — gauche",
    "couple_reference_droite": "Couple de référence banc GMP — droite",
    "couple_reference": "Couple de référence banc GMP — voie unique (si pas de G/D)",
    "regime": "Régime / vitesse de rotation",
    "temperature": "Température (arbre, capteur ou électronique rotor)",
}

# Ordre d'affichage dans l'interface : les deux voies de référence d'abord, la
# voie unique ensuite, puisqu'elle ne sert que de repli.
ORDRE_CANAUX = (
    "couple_mesure_gauche",
    "couple_mesure_droite",
    "couple_reference_gauche",
    "couple_reference_droite",
    "couple_reference",
    "regime",
    "temperature",
)

LIBELLES_TYPES = {
    "balayage": "Balayage — points stabilisés croissants puis décroissants",
    "repetabilite": "Répétabilité — points stabilisés répétés",
    "dynamique": "Dynamique — cycle ou transitoire",
}

# Motifs de proposition du type d'essai à partir du nom de dossier. Ce sont des
# PROPOSITIONS à valider dans l'interface, jamais des décisions : un nom de
# dossier ne dit pas de façon fiable ce qui a été fait pendant l'essai.
MOTIFS_TYPE: tuple[tuple[str, str], ...] = (
    (r"balayage|sweep|montee|escalier", "balayage"),
    (r"statique|stabilis|cpc|palier", "repetabilite"),
    (r"wltc|dynamique|pente|decollage|_da\b|depart|transitoire|cycle", "dynamique"),
)


def _sans_accent(texte: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", texte) if unicodedata.category(c) != "Mn"
    ).lower()


def type_propose(nom_dossier: str) -> str:
    """Propose un type d'essai d'après le nom du dossier.

    Simple aide à la saisie : la valeur doit être confirmée dans l'interface.
    En l'absence de motif reconnu, `dynamique` est proposé — c'est le traitement
    qui ne présuppose pas de points stabilisés, donc le moins susceptible de
    produire des paliers illusoires sur un essai qui n'en comporte pas.
    """
    nom = _sans_accent(nom_dossier)
    for motif, type_essai in MOTIFS_TYPE:
        if re.search(motif, nom):
            return type_essai
    return "dynamique"


def mapping_propose(noms_canaux: list[str]) -> dict[str, Any]:
    """Pré-remplit un mapping avec le premier candidat repéré pour chaque rôle."""
    trouves = candidats(noms_canaux)
    mapping: dict[str, Any] = {}
    for role in CANAUX_SCALAIRES:
        proposes = trouves.get(role) or []
        mapping[role] = proposes[0] if proposes else None
    mapping["etat"] = list(trouves.get("etat") or [])
    return mapping


def noms_tous_dossiers(inventaire_par_dossier: dict[str, dict]) -> list[str]:
    """Union triée des noms de canaux vus dans l'ensemble des dossiers.

    Sert au mapping commun : quand les libellés sont identiques sur toute la
    campagne — le cas courant —, il n'y a aucune raison de les redéclarer
    dossier par dossier.
    """
    vus: dict[str, None] = {}
    for donnees in inventaire_par_dossier.values():
        for nom in donnees.get("noms", []):
            vus.setdefault(nom, None)
    return sorted(vus)


def etat_initial(inventaire_par_dossier: dict[str, dict]) -> dict[str, Any]:
    """Construit l'état de départ de l'interface à partir de l'inventaire."""
    canaux: dict[str, dict] = {}
    essais: dict[str, dict] = {}
    for dossier, donnees in inventaire_par_dossier.items():
        canaux[dossier] = mapping_propose(donnees.get("noms", []))
        essais[dossier] = {
            "type": type_propose(dossier),
            "ligne_droite": False,
            "groupe_remontage": None,
        }
    return {
        "canaux": canaux,
        "essais": essais,
        "canaux_communs": mapping_propose(noms_tous_dossiers(inventaire_par_dossier)),
    }


def _nettoyer_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    """Un canal marqué absent devient `null` — jamais une chaîne vide."""
    propre: dict[str, Any] = {}
    for role in CANAUX_SCALAIRES:
        valeur = mapping.get(role)
        propre[role] = None if valeur in (None, "", CANAL_ABSENT) else valeur
    propre["etat"] = [c for c in (mapping.get("etat") or []) if c and c != CANAL_ABSENT]
    return propre


def construire_dict(
    racine: str,
    dossier_sortie: str,
    pleine_echelle_Nm: float,
    mode_comparaison: str,
    rapport_reduction: float,
    incertitude_reference_k1_Nm: float | None,
    frequence_Hz: float,
    remontage_realise: bool | None,
    canaux: dict[str, dict],
    essais: dict[str, dict],
    paliers: dict[str, Any],
    intercorrelation: dict[str, Any],
    zero: dict[str, Any],
    thermique: dict[str, Any],
    diagnostic: dict[str, Any],
    canaux_communs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble le dictionnaire de configuration, au format du YAML de référence.

    Les canaux marqués absents dans l'interface deviennent `null`, ce qui rendra
    explicitement non calculables les grandeurs qui en dépendent.

    `canaux_communs` renseigné produit un mapping unique dans `canaux.defaut`,
    valable pour tous les essais — le cas courant, où les libellés sont
    identiques sur toute la campagne. `canaux` (par dossier) ne sert que lorsque
    les libellés diffèrent d'un dossier à l'autre.
    """
    if canaux_communs is not None:
        defaut = _nettoyer_mapping(canaux_communs)
        canaux_nettoyes: dict[str, dict] = {}
    else:
        defaut = {}
        canaux_nettoyes = {d: _nettoyer_mapping(m) for d, m in canaux.items()}

    essais_nettoyes: dict[str, dict] = {}
    for dossier, declaration in essais.items():
        type_essai = declaration.get("type")
        if type_essai not in TYPES_ESSAI:
            raise ValueError(
                f"Essai « {dossier} » : type « {type_essai} » invalide "
                f"(attendu : {sorted(TYPES_ESSAI)})."
            )
        groupe = declaration.get("groupe_remontage")
        # Une étiquette réduite à des espaces vaut une étiquette absente : sans
        # ce nettoyage elle compterait comme un groupe de remontage à part
        # entière et fausserait la comparaison avant/après.
        if isinstance(groupe, str):
            groupe = groupe.strip() or None
        essais_nettoyes[dossier] = {
            "type": type_essai,
            "ligne_droite": bool(declaration.get("ligne_droite", False)),
            "groupe_remontage": groupe or None,
        }

    return {
        "racine_donnees": racine,
        "dossier_sortie": dossier_sortie,
        "pleine_echelle_Nm": float(pleine_echelle_Nm),
        "comparaison": {
            "mode": mode_comparaison,
            "rapport_reduction": float(rapport_reduction),
            "incertitude_reference_k1_Nm": incertitude_reference_k1_Nm,
        },
        "acquisition": {"frequence_reechantillonnage_Hz": float(frequence_Hz)},
        "remontage": {"realise": remontage_realise},
        "canaux": {"defaut": defaut, "par_dossier": canaux_nettoyes},
        "essais": essais_nettoyes,
        "paliers": dict(paliers),
        "intercorrelation": dict(intercorrelation),
        "zero": dict(zero),
        "thermique": dict(thermique),
        "diagnostic": dict(diagnostic),
    }


def vers_yaml(configuration: dict[str, Any]) -> str:
    """Sérialise la configuration, dans un YAML relisible et ré-exécutable en CLI."""
    entete = (
        "# Configuration produite par l'interface graphique.\n"
        "# Réutilisable telle quelle en ligne de commande :\n"
        "#   python scripts/02_analyse.py --config <ce fichier>\n\n"
    )
    return entete + yaml.safe_dump(
        configuration, allow_unicode=True, sort_keys=False, default_flow_style=False
    )


def depuis_yaml(chemin: str | Path) -> dict[str, Any]:
    """Relit une configuration existante pour repeupler l'interface."""
    with open(chemin, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def etat_depuis_dict(brut: dict[str, Any]) -> dict[str, Any]:
    """Extrait de la configuration l'état des widgets (canaux et essais).

    Le mapping `defaut` est rabattu sur chaque dossier : l'interface travaille
    dossier par dossier, sans notion d'héritage, ce qui évite d'afficher une
    valeur dont l'origine serait invisible à l'écran.
    """
    canaux = brut.get("canaux") or {}
    defaut = canaux.get("defaut") or {}
    par_dossier = canaux.get("par_dossier") or {}
    essais = brut.get("essais") or {}

    etat_canaux: dict[str, dict] = {}
    for dossier in set(par_dossier) | set(essais):
        fusion = dict(defaut)
        fusion.update(par_dossier.get(dossier) or {})
        mapping = {role: fusion.get(role) for role in CANAUX_SCALAIRES}
        etat = fusion.get("etat") or []
        mapping["etat"] = [etat] if isinstance(etat, str) else list(etat)
        etat_canaux[dossier] = mapping

    etat_essais = {
        dossier: {
            "type": (spec or {}).get("type", type_propose(dossier)),
            "ligne_droite": bool((spec or {}).get("ligne_droite", False)),
            "groupe_remontage": (spec or {}).get("groupe_remontage"),
        }
        for dossier, spec in essais.items()
    }
    return {"canaux": etat_canaux, "essais": etat_essais}


def scalaires_depuis_dict(brut: dict[str, Any]) -> dict[str, Any]:
    """Extrait de la configuration les valeurs scalaires de l'interface.

    Sert à repeupler l'écran depuis un YAML enregistré, pour qu'un
    rafraîchissement de page ne coûte jamais le travail de saisie.
    """
    comparaison = brut.get("comparaison") or {}
    thermique = brut.get("thermique") or {}
    u_ref = comparaison.get("incertitude_reference_k1_Nm")
    plage = thermique.get("plage_service_C")
    realise = (brut.get("remontage") or {}).get("realise")

    valeurs: dict[str, Any] = {
        "racine": str(brut.get("racine_donnees") or ""),
        "dossier_sortie": str(brut.get("dossier_sortie") or "sortie"),
        "pe": float(brut.get("pleine_echelle_Nm") or 1500.0),
        "mode": comparaison.get("mode", "moyenne"),
        "rapport": float(comparaison.get("rapport_reduction", 1.0)),
        "freq": float((brut.get("acquisition") or {}).get("frequence_reechantillonnage_Hz", 100.0)),
        "u_ref_connue": u_ref is not None,
        "u_ref": float(u_ref) if u_ref is not None else 2.0,
        "plage_connue": plage is not None,
        "plage": float(plage) if plage is not None else 40.0,
        "remontage": {True: "Oui", False: "Non", None: "Non précisé"}[realise],
    }
    for section, prefixe in (
        ("paliers", "p"), ("intercorrelation", "i"), ("zero", "z"),
        ("thermique", "t"), ("diagnostic", "d"),
    ):
        for cle, valeur in (brut.get(section) or {}).items():
            valeurs[f"{prefixe}::{cle}"] = valeur
    return valeurs


def valider(configuration: dict[str, Any]) -> tuple[Config | None, list[str]]:
    """Tente de construire la configuration ; renvoie (config, erreurs).

    Ne lève pas : l'interface doit pouvoir afficher l'erreur plutôt que planter.
    """
    try:
        return Config.depuis_dict(configuration), []
    except Exception as exc:
        return None, [str(exc)]
