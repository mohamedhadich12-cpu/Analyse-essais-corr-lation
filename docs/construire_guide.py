#!/usr/bin/env python3
"""Construit le guide utilisateur PDF depuis sa source HTML.

    python docs/construire_guide.py

Étapes :
  1. détourage automatique des marges vides des captures d'écran ;
  2. incorporation des images en data URI — le HTML devient autonome et le
     rendu ne dépend plus d'aucun chemin relatif ;
  3. rendu PDF par Chromium (moteur d'impression), format A4 avec pied de page.

Prérequis : `playwright` et un Chromium installé
(`python -m pip install playwright pillow numpy && python -m playwright install chromium`).

Pour rafraîchir les captures d'écran après une évolution de l'interface, voir
`docs/source/captures.py`.
"""

from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parent / "source"
HTML = SOURCE / "guide.html"
IMAGES = SOURCE / "img"
SORTIE = Path(__file__).resolve().parent / "Guide_correlation_couple_banc_GMP.pdf"

PIED = (
    '<div style="width:100%;font-family:sans-serif;font-size:7.5pt;color:#898781;'
    'padding:0 15mm;display:flex;justify-content:space-between;">'
    "<span>Corrélation couple transmissions / banc GMP — guide d'utilisation</span>"
    '<span class="pageNumber"></span></div>'
)


def detourer(chemin: Path, marge: int = 24) -> None:
    """Retire les marges uniformes autour du contenu utile d'une capture.

    Le fond de l'interface est quasi uni : toute ligne ou colonne dont tous les
    pixels sont à la couleur du coin est du vide. Sans ce détourage, une capture
    haute ne tient pas dans une page et se retrouve réduite au point d'être
    illisible.
    """
    import numpy as np
    from PIL import Image

    image = Image.open(chemin).convert("RGB")
    tableau = np.asarray(image).astype(int)
    utile = np.abs(tableau - tableau[0, 0]).sum(axis=2) > 12
    lignes = np.flatnonzero(utile.any(axis=1))
    colonnes = np.flatnonzero(utile.any(axis=0))
    if not lignes.size or not colonnes.size:
        return
    hauteur, largeur = tableau.shape[:2]
    boite = (
        max(0, colonnes[0] - marge), max(0, lignes[0] - marge),
        min(largeur, colonnes[-1] + marge), min(hauteur, lignes[-1] + marge),
    )
    image.crop(boite).save(chemin)


def incorporer(html: str) -> str:
    """Remplace chaque `src="img/…"` par l'image encodée en base64."""
    def remplacer(correspondance: re.Match) -> str:
        chemin = IMAGES / Path(correspondance.group(1)).name
        donnees = base64.b64encode(chemin.read_bytes()).decode()
        return f'src="data:image/png;base64,{donnees}"'

    return re.sub(r'src="(img/[^"]+)"', remplacer, html)


def rendre(html_autonome: Path, sortie: Path) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pilote:
        navigateur = pilote.chromium.launch()
        page = navigateur.new_page()
        page.goto(html_autonome.resolve().as_uri(), wait_until="load")
        page.wait_for_timeout(2500)
        page.pdf(
            path=str(sortie), format="A4", print_background=True,
            margin={"top": "17mm", "bottom": "18mm", "left": "15mm", "right": "15mm"},
            display_header_footer=True, header_template="<div></div>",
            footer_template=PIED,
        )
        navigateur.close()


def main() -> int:
    if not HTML.is_file():
        print(f"Source introuvable : {HTML}", file=sys.stderr)
        return 2
    for image in sorted(IMAGES.glob("*.png")):
        detourer(image)
    autonome = SOURCE / "guide_autonome.html"
    autonome.write_text(incorporer(HTML.read_text(encoding="utf-8")), encoding="utf-8")
    rendre(autonome, SORTIE)
    autonome.unlink()
    print(f"Guide écrit : {SORTIE}  ({SORTIE.stat().st_size / 1e6:.2f} Mo)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
