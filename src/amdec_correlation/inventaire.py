"""Étape 1 — inventaire des canaux présents dans les acquisitions.

Produit, pour un échantillon de fichiers de chaque sous-dossier d'essai :

  * un rapport Markdown listant tous les canaux, leurs unités, leur cadence et
    leur plage de temps (c'est ce document qui permet de **confirmer les
    libellés** avant de renseigner le mapping) ;
  * un CSV à plat de tous les canaux vus, pour recherche rapide ;
  * un squelette YAML de mapping, pré-rempli avec des **candidats** repérés par
    mots-clés.

Les candidats sont des propositions explicitement marquées « À VÉRIFIER », pas
des hypothèses appliquées : rien n'est utilisé en analyse tant que le mapping
n'a pas été confirmé à la main.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .io_mdf import DescriptionCanal
from .lecteurs import decrire_canaux, lister_fichiers

# Motifs de repérage des candidats. Volontairement larges : mieux vaut proposer
# plusieurs candidats à trancher que de rater le bon canal.
#
# `exclut` est indispensable pour séparer les voies mesurées des voies de
# référence : sans lui, un canal nommé « Trq_Ref_BancGMP_G » serait proposé à la
# fois comme couple mesuré gauche et comme référence gauche.
COUPLE = r"(couple|torque|trq|tq|cpl)"
REFERENCE = r"(ref|banc|bench|gmp|consigne|cible|target)"
GAUCHE = r"(gauche|left|\bg\b|_g_|_g$|_l$|lh)"
DROITE = r"(droit|right|\bd\b|_d_|_d$|_r$|rh)"

MOTIFS: dict[str, dict[str, tuple[str, ...]]] = {
    "couple_mesure_gauche": {"inclut": (COUPLE, GAUCHE), "exclut": (REFERENCE,)},
    "couple_mesure_droite": {"inclut": (COUPLE, DROITE), "exclut": (REFERENCE,)},
    "couple_reference_gauche": {"inclut": (COUPLE, REFERENCE, GAUCHE), "exclut": ()},
    "couple_reference_droite": {"inclut": (COUPLE, REFERENCE, DROITE), "exclut": ()},
    # Référence en voie unique : on écarte ce qui porte déjà une latéralité.
    "couple_reference": {"inclut": (COUPLE, REFERENCE), "exclut": (GAUCHE, DROITE)},
    "regime": {"inclut": (r"(regime|rpm|speed|vitesse|omega|roue|wheel|tr[_/ ]?min|^n[_ ]|[_ ]n$)",),
               "exclut": ()},
    "temperature": {"inclut": (r"(temp|tmp|therm|degc|\bt°|^t[_ ]|[_ ]t[_ ])",), "exclut": ()},
    "etat": {"inclut": (r"(led|crc|err|erreur|fault|status|statut|etat|state|diag|qualit|valid)",),
             "exclut": ()},
}


def _correspond(nom: str, motifs: dict[str, tuple[str, ...]]) -> bool:
    n = nom.lower()
    if any(re.search(m, n) for m in motifs["exclut"]):
        return False
    return all(re.search(m, n) for m in motifs["inclut"])


def candidats(noms: list[str]) -> dict[str, list[str]]:
    """Propose, par nom logique, les canaux dont le libellé colle aux motifs."""
    trouves: dict[str, list[str]] = {cle: [] for cle in MOTIFS}
    for nom in noms:
        for cle, motifs in MOTIFS.items():
            if _correspond(nom, motifs):
                trouves[cle].append(nom)
    # Un canal de couple non identifié comme gauche/droite/référence reste utile
    # à signaler : il figurera dans la liste complète du rapport.
    return trouves


@dataclass
class InventaireDossier:
    dossier: str
    fichiers_examines: list[Path]
    n_fichiers_total: int
    canaux: dict[str, list[DescriptionCanal]]  # nom fichier -> descriptions
    erreurs: list[str]

    @property
    def noms_uniques(self) -> list[str]:
        vus: dict[str, None] = {}
        for descriptions in self.canaux.values():
            for d in descriptions:
                vus.setdefault(d.nom, None)
        return list(vus)

    @property
    def libelles_coherents(self) -> bool:
        """Vrai si tous les fichiers examinés portent le même jeu de canaux."""
        jeux = [frozenset(d.nom for d in desc) for desc in self.canaux.values()]
        return len(set(jeux)) <= 1


NOM_RACINE = "(racine)"


def inventorier(
    racine: Path, sous_dossiers: list[str] | None = None, n_fichiers: int = 2
) -> list[InventaireDossier]:
    """Inventorie un échantillon d'acquisitions par groupe.

    Un groupe est un sous-dossier d'essai — organisation par type — mais aussi
    la **racine elle-même** lorsqu'elle contient directement des acquisitions.
    Sans cela, un dossier plat, qui est l'organisation la plus courante, ne
    produirait aucun inventaire et donc aucun mapping.
    """
    racine = Path(racine)
    if not racine.is_dir():
        raise FileNotFoundError(f"Racine des données introuvable : {racine}")

    groupes: list[tuple[str, list[Path]]] = []
    if sous_dossiers:
        groupes = [(nom, lister_fichiers(racine / nom)) for nom in sous_dossiers]
    else:
        directs = [f for f in lister_fichiers(racine) if f.parent == racine]
        if directs:
            groupes.append((NOM_RACINE, directs))
        groupes += [
            (p.name, lister_fichiers(p))
            for p in sorted(racine.iterdir(), key=lambda p: p.name.lower())
            if p.is_dir() and lister_fichiers(p, 1)
        ]

    resultats: list[InventaireDossier] = []
    for nom, tous in groupes:
        echantillon = tous[:n_fichiers]
        canaux: dict[str, list[DescriptionCanal]] = {}
        erreurs: list[str] = []
        for fichier in echantillon:
            try:
                canaux[fichier.name] = decrire_canaux(fichier)
            except Exception as exc:
                erreurs.append(f"{fichier.name} : {exc}")
        resultats.append(
            InventaireDossier(nom, echantillon, len(tous), canaux, erreurs)
        )
    return resultats


# ---------------------------------------------------------------------------
# Restitutions
# ---------------------------------------------------------------------------


def rapport_markdown(inventaires: list[InventaireDossier], racine: Path) -> str:
    blocs = [
        "# Étape 1 — Inventaire des canaux MDF4",
        "",
        f"Racine explorée : `{racine}`",
        "",
        "Ce document sert à **confirmer les libellés exacts** avant de renseigner le mapping.",
        "Les « candidats » ci-dessous sont proposés par simple correspondance de mots-clés :",
        "ils doivent être validés un par un, aucun n'est utilisé tant qu'il n'a pas été",
        "recopié dans `config/correlation.yaml`.",
        "",
    ]
    for inv in inventaires:
        blocs.append(f"## {inv.dossier}")
        blocs.append("")
        blocs.append(
            f"{inv.n_fichiers_total} fichier(s) `.mf4` — "
            f"{len(inv.fichiers_examines)} examiné(s) : "
            + (", ".join(f"`{f.name}`" for f in inv.fichiers_examines) or "_aucun_")
        )
        blocs.append("")
        if inv.erreurs:
            blocs.append("**Erreurs de lecture :**")
            blocs.extend(f"- {e}" for e in inv.erreurs)
            blocs.append("")
        if not inv.canaux:
            blocs.append("_Aucun canal lisible dans ce dossier._\n")
            continue
        if not inv.libelles_coherents:
            blocs.append(
                "> ⚠️ Les fichiers examinés de ce dossier **ne portent pas le même jeu de "
                "canaux**. Le mapping devra être vérifié fichier par fichier.\n"
            )

        propositions = candidats(inv.noms_uniques)
        blocs.append("**Candidats repérés (À VÉRIFIER) :**")
        blocs.append("")
        blocs.append("| Rôle | Candidat(s) |")
        blocs.append("|---|---|")
        for cle, noms in propositions.items():
            valeur = ", ".join(f"`{n}`" for n in noms) if noms else "_aucun candidat_"
            blocs.append(f"| {cle} | {valeur} |")
        blocs.append("")

        for fichier, descriptions in inv.canaux.items():
            blocs.append(f"<details><summary>Canaux de <code>{fichier}</code> "
                         f"({len(descriptions)})</summary>")
            blocs.append("")
            blocs.append("| Canal | Unité | Groupe | n | Fréq. (Hz) | t début (s) | t fin (s) |")
            blocs.append("|---|---|---|---|---|---|---|")
            for d in descriptions:
                freq = f"{d.frequence_Hz:.2f}" if d.frequence_Hz else "—"
                t0 = f"{d.t_debut:.3f}" if d.t_debut is not None else "—"
                t1 = f"{d.t_fin:.3f}" if d.t_fin is not None else "—"
                blocs.append(
                    f"| `{d.nom}` | {d.unite or '—'} | {d.groupe} | {d.n_echantillons} | "
                    f"{freq} | {t0} | {t1} |"
                )
            blocs.append("")
            blocs.append("</details>")
            blocs.append("")

            # Diagnostic de base de temps : autant de groupes = autant d'horloges.
            groupes = defaultdict(list)
            for d in descriptions:
                groupes[d.groupe].append(d)
            if len(groupes) > 1:
                details = " ; ".join(
                    f"groupe {g} : {v[0].n_echantillons} pts à "
                    f"{v[0].frequence_Hz:.1f} Hz" if v[0].frequence_Hz else f"groupe {g}"
                    for g, v in sorted(groupes.items())
                )
                blocs.append(
                    f"> Base de temps : **{len(groupes)} groupes** de canaux, donc autant de "
                    f"canaux maîtres distincts ({details}). Un rééchantillonnage sur grille "
                    f"commune est nécessaire — le pipeline le fait et le documente.\n"
                )
            else:
                blocs.append("> Base de temps : un seul groupe, tous les canaux partagent le même maître.\n")
    return "\n".join(blocs) + "\n"


def ecrire_csv(inventaires: list[InventaireDossier], chemin: Path) -> Path:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with open(chemin, "w", newline="", encoding="utf-8-sig") as fh:
        redacteur = csv.writer(fh, delimiter=";")
        redacteur.writerow(
            ["dossier", "fichier", "canal", "unite", "groupe", "n_echantillons",
             "frequence_Hz", "t_debut_s", "t_fin_s", "commentaire"]
        )
        for inv in inventaires:
            for fichier, descriptions in inv.canaux.items():
                for d in descriptions:
                    redacteur.writerow(
                        [inv.dossier, fichier, d.nom, d.unite, d.groupe, d.n_echantillons,
                         f"{d.frequence_Hz:.4f}" if d.frequence_Hz else "",
                         f"{d.t_debut:.4f}" if d.t_debut is not None else "",
                         f"{d.t_fin:.4f}" if d.t_fin is not None else "",
                         d.commentaire]
                    )
    return chemin


def squelette_yaml(inventaires: list[InventaireDossier], racine: Path) -> str:
    """Squelette de mapping pré-rempli avec les candidats, à valider à la main."""
    roles = (
        "couple_mesure_gauche", "couple_mesure_droite",
        "couple_reference", "couple_reference_gauche", "couple_reference_droite",
        "regime", "temperature",
    )
    lignes = [
        "# Mapping des canaux — SQUELETTE À VALIDER",
        "#",
        "# Généré par l'inventaire : chaque valeur est un CANDIDAT repéré par mot-clé,",
        "# pas une donnée confirmée. Vérifier chaque ligne dans le rapport d'inventaire,",
        "# corriger si besoin, et mettre `null` pour tout canal réellement absent.",
        "#",
        "# À recopier dans la section `canaux:` de config/correlation.yaml.",
        "",
        "canaux:",
        "  defaut:",
        *[f"    {role}: null" for role in roles],
        "    etat: []",
        "",
        "  par_dossier:",
    ]
    for inv in inventaires:
        lignes.append(f'    "{inv.dossier}":')
        if not inv.canaux:
            lignes.append("      # aucun canal lisible dans ce dossier")
            lignes.append("      {}")
            continue
        propositions = candidats(inv.noms_uniques)
        for cle in roles:
            noms = propositions[cle]
            if noms:
                autres = f"  # AUTRES CANDIDATS : {', '.join(noms[1:])}" if len(noms) > 1 else ""
                lignes.append(f'      {cle}: "{noms[0]}"  # À VÉRIFIER{autres}')
            else:
                lignes.append(f"      {cle}: null  # aucun candidat repéré")
        etats = propositions["etat"]
        if etats:
            liste = ", ".join(f'"{n}"' for n in etats)
            lignes.append(f"      etat: [{liste}]  # À VÉRIFIER")
        else:
            lignes.append("      etat: []")
        lignes.append("")
    return "\n".join(lignes) + "\n"


def executer(
    racine: Path, dossier_sortie: Path, sous_dossiers: list[str] | None = None,
    n_fichiers: int = 2,
) -> dict[str, Path]:
    """Lance l'inventaire complet et écrit les trois restitutions."""
    inventaires = inventorier(racine, sous_dossiers, n_fichiers)
    dossier_sortie.mkdir(parents=True, exist_ok=True)
    md = dossier_sortie / "inventaire_canaux.md"
    md.write_text(rapport_markdown(inventaires, racine), encoding="utf-8")
    csv_path = ecrire_csv(inventaires, dossier_sortie / "canaux.csv")
    yml = dossier_sortie / "canaux_proposition.yaml"
    yml.write_text(squelette_yaml(inventaires, racine), encoding="utf-8")
    return {"markdown": md, "csv": csv_path, "yaml": yml}
