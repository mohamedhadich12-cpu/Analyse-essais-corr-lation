#!/usr/bin/env python3
"""Étapes 2 et 3 — traitements, rapport Markdown et figures PNG.

    python scripts/02_analyse.py --config config/correlation.yaml

Prérequis : le mapping des canaux doit avoir été validé dans le fichier de
configuration (cf. `scripts/01_inventaire.py`).

Aucun fichier `.mf4` n'est modifié : ouverture en lecture seule uniquement.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amdec_correlation.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["analyse", *sys.argv[1:]]))
