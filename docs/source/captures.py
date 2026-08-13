#!/usr/bin/env python3
"""Rafraîchit les captures d'écran du guide utilisateur.

À relancer après toute évolution visible de l'interface, sinon le guide décrit
un écran qui n'existe plus.

Mode d'emploi, depuis la racine du projet :

    1. générer un jeu de données de démonstration :
       python tests/generer_mf4_synthetique.py --sortie donnees_synthetiques

    2. lancer l'interface DEPUIS LA RACINE, pour que `.streamlit/config.toml`
       s'applique — sans quoi les captures montreraient le thème par défaut de
       Streamlit et non celui du projet :
       python -m streamlit run src/amdec_correlation/interface/app.py --server.port 8511

    3. dans une autre fenêtre :
       python docs/source/captures.py

    4. reconstruire le PDF :
       python docs/construire_guide.py

Prérequis : `python -m pip install playwright && python -m playwright install chromium`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

RACINE = Path(__file__).resolve().parents[2]
IMAGES = Path(__file__).resolve().parent / "img"

# Chromium déjà installé ailleurs que dans le cache Playwright : renseigner
# CHROMIUM_EXECUTABLE plutôt que relancer `playwright install`.
CHROMIUM = os.environ.get("CHROMIUM_EXECUTABLE") or None


def capturer(url: str, racine_donnees: Path, dossier_sortie: Path) -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pilote:
        navigateur = pilote.chromium.launch(executable_path=CHROMIUM)
        # Facteur 2 : les captures restent nettes une fois réduites dans le PDF.
        page = navigateur.new_page(
            viewport={"width": 1440, "height": 1000}, device_scale_factor=2
        )
        page.goto(url, wait_until="networkidle", timeout=90_000)
        page.wait_for_timeout(4000)

        champs = page.locator('section[data-testid="stSidebar"] input[type="text"]')
        champs.nth(0).fill(str(racine_donnees)); champs.nth(0).press("Enter")
        page.wait_for_timeout(2500)
        champs.nth(1).fill(str(dossier_sortie)); champs.nth(1).press("Enter")
        page.wait_for_timeout(2500)
        page.get_by_role("button", name="Inventorier les canaux").click()
        page.wait_for_timeout(11_000)

        zone = page.locator("div.block-container").first

        def onglet(nom: str, attente: int = 3000) -> None:
            page.get_by_role("tab", name=nom).click()
            page.wait_for_timeout(attente)

        onglet("1 · Exploration")
        page.get_by_text("1-Balayage Couple — 1 fichier(s)").click()
        page.wait_for_timeout(2500)
        zone.screenshot(path=str(IMAGES / "01_exploration.png"))

        onglet("3 · Canaux", 3500); zone.screenshot(path=str(IMAGES / "02_canaux.png"))

        # La détection lit toutes les acquisitions : c'est la capture la plus lente.
        onglet("4 · Zones détectées")
        page.get_by_role("button", name="Détecter les zones").click()
        page.wait_for_timeout(25_000)
        # Les figures sont repliées pour la capture : le guide les montre juste
        # après, en pleine résolution. Dépliées, elles pousseraient le tableau
        # hors de la fenêtre et la capture serait tronquée.
        page.get_by_text("Où se trouvent ces zones").click()
        page.wait_for_timeout(1500)
        zone.screenshot(path=str(IMAGES / "03_zones.png"))

        onglet("5 · Hypothèses");   zone.screenshot(path=str(IMAGES / "04_hypotheses.png"))

        # Deux combinaisons : la capture doit montrer qu'elles partagent le tracé.
        onglet("2 · Visualisation", 5000)
        combinaisons = page.get_by_role("spinbutton", name="Nombre de combinaisons")
        combinaisons.fill("2"); combinaisons.press("Enter")
        page.wait_for_timeout(4000)
        for libelle, valeur in (
            ("Canal A — combinaison 1", "Trq_Transmission_G"),
            ("Canal B — combinaison 1", "Trq_Transmission_D"),
            ("Canal A — combinaison 2", "Trq_Ref_BancGMP_G"),
            ("Canal B — combinaison 2", "Trq_Ref_BancGMP_D"),
        ):
            page.get_by_role("combobox", name=libelle).click()
            page.wait_for_timeout(500)
            page.get_by_role("option", name=valeur, exact=True).first.click()
            page.wait_for_timeout(1200)
        # Fenêtre agrandie le temps de la capture : une capture d'élément est
        # tronquée à la hauteur de la fenêtre, et cet onglet — deux combinaisons
        # plus le tracé — dépasse la hauteur de travail.
        page.set_viewport_size({"width": 1440, "height": 1360})
        page.wait_for_timeout(5000)
        zone.screenshot(path=str(IMAGES / "10_visualisation.png"))
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.wait_for_timeout(1500)

        onglet("6 · Analyse & résultats", 3500)
        page.get_by_role("button", name="Lancer l'analyse").click()
        page.wait_for_timeout(50_000)
        zone.screenshot(path=str(IMAGES / "05_resultats.png"))
        onglet("Cartes de contrôle");  zone.screenshot(path=str(IMAGES / "07_spc.png"))
        navigateur.close()

    # Une vraie figure d'analyse vaut mieux qu'une capture de l'onglet Figures.
    figures = dossier_sortie / "figures"
    zones = sorted(figures.glob("zones_*balayage*.png")) or sorted(figures.glob("zones_*.png"))
    a_copier = [(figures / "regression_paliers.png", "08_figure_exemple.png")]
    if zones:
        a_copier.append((zones[0], "11_zones_figure.png"))
    for source, cible in a_copier:
        if source.is_file():
            (IMAGES / cible).write_bytes(source.read_bytes())
    print(f"Captures écrites dans {IMAGES}")


if __name__ == "__main__":
    parseur = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parseur.add_argument("--url", default="http://localhost:8511")
    parseur.add_argument("--donnees", default=str(RACINE / "donnees_synthetiques"))
    parseur.add_argument("--sortie", default=str(RACINE / "sortie_guide"))
    args = parseur.parse_args()
    capturer(args.url, Path(args.donnees), Path(args.sortie))
