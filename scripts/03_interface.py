#!/usr/bin/env python3
"""Lance l'interface graphique locale.

    python scripts/03_interface.py

Ouvre une page dans le navigateur. Tout s'exécute sur le poste : les
acquisitions ne transitent par aucun réseau, et les `.mf4` sont ouverts en
lecture seule.

Options utiles :
    --port 8502        si le port par défaut est déjà pris
    --sans-navigateur  n'ouvre pas le navigateur automatiquement
"""

import argparse
import subprocess
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
APPLICATION = RACINE / "src" / "amdec_correlation" / "interface" / "app.py"


def main() -> int:
    parseur = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parseur.add_argument("--port", type=int, default=8501)
    parseur.add_argument("--sans-navigateur", action="store_true")
    args = parseur.parse_args()

    try:
        import streamlit  # noqa: F401
    except ImportError:
        print(
            "Streamlit n'est pas installé. Installe-le avec :\n"
            "    python -m pip install streamlit\n"
            "(ou : python -m pip install -r requirements.txt)",
            file=sys.stderr,
        )
        return 2

    commande = [
        sys.executable, "-m", "streamlit", "run", str(APPLICATION),
        "--server.port", str(args.port),
        # L'interface est strictement locale : pas de collecte d'usage, pas
        # d'écoute au-delà de la machine.
        "--browser.gatherUsageStats", "false",
        "--server.address", "localhost",
    ]
    if args.sans_navigateur:
        commande += ["--server.headless", "true"]

    print(f"Interface disponible sur  http://localhost:{args.port}")
    return subprocess.call(commande, cwd=RACINE)


if __name__ == "__main__":
    raise SystemExit(main())
