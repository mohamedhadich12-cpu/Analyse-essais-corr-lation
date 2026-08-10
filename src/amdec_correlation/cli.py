"""Interface en ligne de commande.

    python -m amdec_correlation inventaire --racine "C:/.../Documents"
    python -m amdec_correlation analyse --config config/correlation.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import analyse as A
from . import inventaire as I
from . import rapport as R
from .config import Config, ErreurConfig


def _cmd_inventaire(args: argparse.Namespace) -> int:
    sorties = I.executer(
        racine=Path(args.racine),
        dossier_sortie=Path(args.sortie),
        sous_dossiers=args.dossiers or None,
        n_fichiers=args.n_fichiers,
    )
    print("Inventaire terminé :")
    for cle, chemin in sorties.items():
        print(f"  - {cle:9s} : {chemin}")
    print(
        "\nÉtape suivante : relire `inventaire_canaux.md`, valider les candidats, puis "
        "recopier la section `canaux:` de `canaux_proposition.yaml` dans "
        "`config/correlation.yaml`."
    )
    return 0


def _cmd_analyse(args: argparse.Namespace) -> int:
    cfg = Config.charger(args.config)
    if args.racine:
        cfg.racine = Path(args.racine)
    if args.sortie:
        cfg.dossier_sortie = Path(args.sortie)

    if not cfg.racine.is_dir():
        print(f"ERREUR : racine des données introuvable : {cfg.racine}", file=sys.stderr)
        return 2

    for avertissement in cfg.verifier_mapping():
        print(f"  [mapping] {avertissement}", file=sys.stderr)
    for avertissement in cfg.verifier_saisies():
        print(f"  [saisie]  {avertissement}", file=sys.stderr)

    campagne = A.analyser(cfg)
    chemin = R.ecrire(campagne)

    print(f"\nRapport écrit : {chemin}")
    figures = campagne.figures + [f for e in campagne.essais for f in e.figures]
    print(f"Figures ({len(figures)}) dans : {cfg.dossier_sortie / 'figures'}")
    for e in campagne.essais:
        etat = f"{len(e.fichiers_traites)} fichier(s)" if e.fichiers_traites else "NON EXPLOITÉ"
        print(f"  - {e.nom:45s} {etat}")
        for erreur in e.erreurs:
            print(f"      ! {erreur}")
    return 0


def construire_parseur() -> argparse.ArgumentParser:
    parseur = argparse.ArgumentParser(
        prog="amdec_correlation",
        description="Corrélation couple transmissions instrumentées / banc GMP (chapitre 10 AMDEC).",
    )
    sous = parseur.add_subparsers(dest="commande", required=True)

    inv = sous.add_parser("inventaire", help="Étape 1 : lister les canaux des acquisitions.")
    inv.add_argument("--racine", required=True, help="Dossier racine contenant les sous-dossiers d'essai.")
    inv.add_argument("--sortie", default="sortie/inventaire", help="Dossier de sortie.")
    inv.add_argument("--dossiers", nargs="*", help="Restreindre à certains sous-dossiers.")
    inv.add_argument("-n", "--n-fichiers", type=int, default=2,
                     help="Nombre de fichiers échantillonnés par sous-dossier (défaut : 2).")
    inv.set_defaults(fonction=_cmd_inventaire)

    ana = sous.add_parser("analyse", help="Étape 2-3 : traitements, rapport et figures.")
    ana.add_argument("--config", default="config/correlation.yaml", help="Fichier de configuration YAML.")
    ana.add_argument("--racine", help="Surcharge la racine des données.")
    ana.add_argument("--sortie", help="Surcharge le dossier de sortie.")
    ana.set_defaults(fonction=_cmd_analyse)
    return parseur


def main(argv: list[str] | None = None) -> int:
    args = construire_parseur().parse_args(argv)
    try:
        return args.fonction(args)
    except (ErreurConfig, FileNotFoundError) as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
