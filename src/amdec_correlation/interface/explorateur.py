"""Ouverture de l'explorateur de fichiers du poste, pour choisir un dossier.

Streamlit s'affiche dans un navigateur web, qui n'a pas le droit d'ouvrir
l'explorateur du système ni de lire un chemin local. Mais le processus Python,
lui, tourne sur le poste de l'utilisateur : c'est donc **lui** qui ouvre la
boîte de dialogue, sur le bureau, et renvoie le chemin retenu.

La boîte est ouverte dans un **processus séparé**, jamais dans celui de
Streamlit. Tkinter veut être piloté depuis le fil principal et n'aime pas être
créé puis détruit à répétition ; le script Streamlit, lui, s'exécute dans un fil
de travail et se relance à chaque interaction. Les mélanger fige l'application
une fois sur deux, et le défaut est difficile à reproduire. Un sous-processus
rend le problème sans objet : il naît, affiche la boîte, écrit le chemin sur sa
sortie standard, et meurt.

Cette approche suppose que le navigateur et Python tournent sur la MÊME machine
— ce qui est le mode d'emploi de l'outil. Si l'application était servie depuis
un poste distant, la boîte s'ouvrirait sur ce poste-là : `disponible()` ne peut
pas le détecter, c'est à l'exploitant de le savoir.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Laisse tout le temps de naviguer, sans jamais figer l'application
# indéfiniment si la boîte est laissée ouverte puis oubliée.
DELAI_MAX_S = 300.0

_SCRIPT = r"""
import sys
import tkinter as tk
from tkinter import filedialog

racine = tk.Tk()
racine.withdraw()
# Sans cela, la boîte s'ouvre DERRIÈRE la fenêtre du navigateur : l'utilisateur
# croit que le bouton n'a rien fait et clique à nouveau.
racine.attributes("-topmost", True)
racine.update()
chemin = filedialog.askdirectory(
    title=sys.argv[1] if len(sys.argv) > 1 else "Choisir un dossier",
    initialdir=sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None,
    mustexist=False,
)
racine.destroy()
sys.stdout.write(chemin or "")
"""


class ExplorateurIndisponible(RuntimeError):
    """L'explorateur graphique ne peut pas être ouvert sur ce poste."""


def disponible() -> tuple[bool, str]:
    """Dit si la boîte de dialogue peut être ouverte, et sinon pourquoi.

    Le diagnostic est rendu en clair : un bouton qui échoue sans expliquer
    laisserait croire à une panne de l'outil, alors qu'il s'agit d'une
    installation de Python sans Tk ou d'une session sans affichage graphique.
    """
    try:
        import tkinter  # noqa: F401
    except Exception as exc:  # pragma: no cover - dépend de l'installation
        return False, (
            f"le module Tk n'est pas disponible dans cet interpréteur ({exc}). "
            "Sous Linux, il s'installe séparément (paquet `python3-tk`)."
        )
    if os.name != "nt" and sys.platform != "darwin" and not os.environ.get("DISPLAY"):
        return False, (
            "aucun affichage graphique n'est accessible depuis ce processus "
            "(variable DISPLAY absente) : l'outil tourne probablement sur une "
            "machine distante ou dans un conteneur."
        )
    return True, ""


def choisir_dossier(depart: str = "", titre: str = "Choisir un dossier") -> str | None:
    """Ouvre l'explorateur du poste et renvoie le dossier choisi.

    Renvoie `None` si l'utilisateur annule. Lève `ExplorateurIndisponible` si la
    boîte ne peut pas être ouverte — jamais une valeur par défaut silencieuse :
    l'exploitant doit savoir qu'il lui faut saisir le chemin à la main.
    """
    ok, motif = disponible()
    if not ok:
        raise ExplorateurIndisponible(motif)

    depart = str(depart or "")
    if depart and not Path(depart).is_dir():
        depart = ""
    try:
        resultat = subprocess.run(
            [sys.executable, "-c", _SCRIPT, titre, depart],
            capture_output=True, text=True, timeout=DELAI_MAX_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise ExplorateurIndisponible(
            f"la fenêtre est restée ouverte plus de {DELAI_MAX_S:.0f} s sans réponse"
        ) from exc
    except Exception as exc:  # pragma: no cover - dépend du poste
        raise ExplorateurIndisponible(str(exc)) from exc

    if resultat.returncode != 0:
        detail = (resultat.stderr or "").strip().splitlines()
        raise ExplorateurIndisponible(detail[-1] if detail else "échec de l'explorateur")

    chemin = (resultat.stdout or "").strip()
    return chemin or None
