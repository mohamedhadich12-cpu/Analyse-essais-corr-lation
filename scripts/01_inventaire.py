#!/usr/bin/env python3
"""Étape 1 — inventaire des canaux, sans installation préalable du paquet.

    python scripts/01_inventaire.py --racine "C:/Users/SD17365/Documents"

Écrit dans `sortie/inventaire/` :
  * `inventaire_canaux.md`      — tous les canaux, unités, cadences, bases de temps ;
  * `canaux.csv`                — le même à plat, pour recherche ;
  * `canaux_proposition.yaml`   — squelette de mapping pré-rempli, À VALIDER.

Aucun fichier `.mf4` n'est modifié : ouverture en lecture seule uniquement.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amdec_correlation.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["inventaire", *sys.argv[1:]]))
