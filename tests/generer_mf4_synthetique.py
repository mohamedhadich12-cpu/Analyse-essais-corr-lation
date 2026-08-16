#!/usr/bin/env python3
"""Génère une arborescence d'essais MDF4 SYNTHÉTIQUES, à vérité connue.

⚠️ CES DONNÉES SONT FABRIQUÉES. Elles servent uniquement à :
  * vérifier l'installation (asammdf, scipy, matplotlib) avant de toucher aux
    acquisitions réelles ;
  * valider que le pipeline retrouve bien des grandeurs dont on connaît la
    valeur exacte (tests unitaires).

Aucun chiffre produit à partir de ces fichiers ne doit figurer dans le rapport.

Défauts injectés (vérité terrain, cf. constantes ci-dessous) :
  * erreur de gain et offset de la chaîne de mesure ;
  * hystérésis montée/descente sur le balayage ;
  * biais de remontage entre les répétitions de l'essai CPC ;
  * retard pur de la voie transmissions sur les essais dynamiques ;
  * sensibilité thermique, avec excursion de température sur le cycle.

Les canaux sont volontairement répartis sur TROIS groupes de cadences
différentes (couples, température, état), afin d'exercer le rééchantillonnage
sur base de temps commune.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from asammdf import MDF, Signal

# --- Vérité terrain ---------------------------------------------------------
PLEINE_ECHELLE = 1500.0
GAIN = 1.012  # erreur de sensibilité de +1,2 %
OFFSET_NM = 6.0  # offset de la chaîne de mesure
HYSTERESIS_NM = 3.0  # écart montée/descente
RETARD_S = 0.040  # 40 ms de retard de la voie transmissions
SENSIBILITE_TH_NM_PAR_C = 0.15  # N·m/°C
ECART_VOIES_NM = 2.0  # écart systématique gauche − droite
BIAIS_REMONTAGE_NM = 4.0  # décalage entre les deux groupes de remontage
BRUIT_NM = 0.8  # écart-type du bruit de mesure
FE_COUPLE = 200.0
FE_TEMPERATURE = 1.0
FE_ETAT = 10.0

# Le banc GMP fournit DEUX voies de référence, une par sortie.
NOMS = {
    "gauche": "Trq_Transmission_G",
    "droite": "Trq_Transmission_D",
    "reference_gauche": "Trq_Ref_BancGMP_G",
    "reference_droite": "Trq_Ref_BancGMP_D",
    "regime": "N_Roue",
    "temperature": "T_Rotor_PCM16",
    "etat": "PCM16_LED_Status",
}


def _ecrire(chemin: Path, t, g, d, ref, regime, t_temp, temperature, t_etat, etat) -> None:
    """Écrit un MDF4 à trois groupes de cadences distinctes.

    Les deux voies de référence portent le même couple : le banc charge ses
    sorties symétriquement. La vérité terrain reste donc inchangée quel que soit
    le mode de comparaison retenu.
    """
    chemin.parent.mkdir(parents=True, exist_ok=True)
    mdf = MDF(version="4.10")
    mdf.append(
        [
            Signal(g, t, name=NOMS["gauche"], unit="N.m"),
            Signal(d, t, name=NOMS["droite"], unit="N.m"),
            Signal(ref, t, name=NOMS["reference_gauche"], unit="N.m"),
            Signal(ref.copy(), t, name=NOMS["reference_droite"], unit="N.m"),
            Signal(regime, t, name=NOMS["regime"], unit="rpm"),
        ],
        comment="Couples et regime",
    )
    mdf.append([Signal(temperature, t_temp, name=NOMS["temperature"], unit="degC")],
               comment="Thermique")
    mdf.append([Signal(etat, t_etat, name=NOMS["etat"], unit="")], comment="Etat telemetrie")
    mdf.save(chemin, overwrite=True)
    mdf.close()


def _annexes(t: np.ndarray, temperature_debut: float, temperature_fin: float):
    """Canaux température (1 Hz) et état (10 Hz), sur leurs propres bases de temps."""
    duree = t[-1]
    t_temp = np.arange(0.0, duree, 1.0 / FE_TEMPERATURE)
    temperature = temperature_debut + (temperature_fin - temperature_debut) * (t_temp / duree)
    t_etat = np.arange(0.0, duree, 1.0 / FE_ETAT)
    etat = np.ones(t_etat.size, dtype=np.uint8)  # 1 = liaison télémétrique nominale
    return t_temp, temperature, t_etat, etat


def _temperature_a(t: np.ndarray, temperature_debut: float, temperature_fin: float) -> np.ndarray:
    return temperature_debut + (temperature_fin - temperature_debut) * (t / t[-1])


def balayage(chemin: Path, graine: int = 0) -> None:
    """Points stabilisés croissants puis décroissants, avec hystérésis."""
    rng = np.random.default_rng(graine)
    niveaux = [0, 150, 300, 450, 600, 750, 900, 1050, 1200]
    sequence = niveaux + niveaux[-2::-1]  # montée puis descente
    duree_palier, duree_rampe = 8.0, 2.0

    t_liste, ref_liste, sens_liste = [], [], []
    horloge = 0.0
    for i, niveau in enumerate(sequence):
        n = int(duree_palier * FE_COUPLE)
        t_liste.append(horloge + np.arange(n) / FE_COUPLE)
        ref_liste.append(np.full(n, float(niveau)))
        sens_liste.append(np.full(n, 1.0 if i < len(niveaux) else -1.0))
        horloge += duree_palier
        if i < len(sequence) - 1:
            m = int(duree_rampe * FE_COUPLE)
            t_liste.append(horloge + np.arange(m) / FE_COUPLE)
            ref_liste.append(np.linspace(niveau, sequence[i + 1], m))
            sens_liste.append(np.full(m, 1.0 if i < len(niveaux) else -1.0))
            horloge += duree_rampe

    t = np.concatenate(t_liste)
    ref = np.concatenate(ref_liste)
    sens = np.concatenate(sens_liste)

    # Température maintenue constante : le balayage teste le gain, l'offset,
    # la non-linéarité et l'hystérésis, sans interférence thermique.
    mesure = GAIN * ref + OFFSET_NM - HYSTERESIS_NM * (sens < 0)
    g = mesure + ECART_VOIES_NM / 2 + rng.normal(0, BRUIT_NM, t.size)
    d = mesure - ECART_VOIES_NM / 2 + rng.normal(0, BRUIT_NM, t.size)
    regime = 50.0 + ref / 20.0

    t_temp, temperature, t_etat, etat = _annexes(t, 20.0, 20.0)
    _ecrire(chemin, t, g, d, ref, regime, t_temp, temperature, t_etat, etat)


def cpc(chemin: Path, graine: int, biais_Nm: float) -> None:
    """Points stabilisés répétés à température stabilisée (20 °C).

    `biais_Nm` reproduit le décalage d'une répétition à l'autre : c'est lui que
    la répétabilité doit mesurer.
    """
    rng = np.random.default_rng(graine)
    niveaux = [0, 200, 400, 600, 800, 1000]
    duree_palier, duree_rampe = 10.0, 2.0

    t_liste, ref_liste = [], []
    horloge = 0.0
    for i, niveau in enumerate(niveaux):
        n = int(duree_palier * FE_COUPLE)
        t_liste.append(horloge + np.arange(n) / FE_COUPLE)
        ref_liste.append(np.full(n, float(niveau)))
        horloge += duree_palier
        if i < len(niveaux) - 1:
            m = int(duree_rampe * FE_COUPLE)
            t_liste.append(horloge + np.arange(m) / FE_COUPLE)
            ref_liste.append(np.linspace(niveau, niveaux[i + 1], m))
            horloge += duree_rampe

    t = np.concatenate(t_liste)
    ref = np.concatenate(ref_liste)
    mesure = GAIN * ref + OFFSET_NM + biais_Nm
    g = mesure + ECART_VOIES_NM / 2 + rng.normal(0, BRUIT_NM, t.size)
    d = mesure - ECART_VOIES_NM / 2 + rng.normal(0, BRUIT_NM, t.size)
    regime = 50.0 + ref / 20.0

    t_temp, temperature, t_etat, etat = _annexes(t, 20.0, 20.0)
    _ecrire(chemin, t, g, d, ref, regime, t_temp, temperature, t_etat, etat)


def dynamique(chemin: Path, graine: int, duree_s: float = 200.0,
              temperature_fin: float = 45.0) -> None:
    """Cycle transitoire : retard pur, excursion thermique, zéros en début et fin."""
    rng = np.random.default_rng(graine)
    t = np.arange(0.0, duree_s, 1.0 / FE_COUPLE)

    # Profil de couple : superposition de composantes lentes et rapides, avec
    # 12 s à couple nul en début et en fin (relevés de zéro).
    ref = (
        420.0
        + 340.0 * np.sin(2 * np.pi * t / 25.0)
        + 190.0 * np.sin(2 * np.pi * t / 7.3 + 1.1)
        + 90.0 * np.sin(2 * np.pi * t / 3.1 + 0.4)
    )
    repos = (t < 12.0) | (t > duree_s - 12.0)
    ref[repos] = 0.0

    temperature = _temperature_a(t, 20.0, temperature_fin)
    # La voie transmissions voit le même couple, retardé, avec gain, offset et
    # dérive thermique.
    ref_retardee = np.interp(t - RETARD_S, t, ref, left=ref[0], right=ref[-1])
    mesure = (
        GAIN * ref_retardee
        + OFFSET_NM
        + SENSIBILITE_TH_NM_PAR_C * (temperature - 20.0)
    )
    g = mesure + ECART_VOIES_NM / 2 + rng.normal(0, BRUIT_NM, t.size)
    d = mesure - ECART_VOIES_NM / 2 + rng.normal(0, BRUIT_NM, t.size)
    regime = np.abs(ref) / 3.0
    regime[repos] = 0.0

    t_temp, temperature_1hz, t_etat, etat = _annexes(t, 20.0, temperature_fin)
    _ecrire(chemin, t, g, d, ref, regime, t_temp, temperature_1hz, t_etat, etat)


ARBORESCENCE = {
    "1-Balayage Couple": [("balayage_01.mf4", lambda p: balayage(p, graine=1))],
    "2-CPC 20°C": [
        ("cpc_rep01.mf4", lambda p: cpc(p, graine=11, biais_Nm=0.0)),
        ("cpc_rep02.mf4", lambda p: cpc(p, graine=12, biais_Nm=1.2)),
        ("cpc_rep03.mf4", lambda p: cpc(p, graine=13, biais_Nm=-0.9)),
    ],
    # Second groupe de remontage : c'est la présence de DEUX groupes distincts
    # qui rend la « répétabilité après remontage » calculable.
    "2-CPC 20°C apres remontage": [
        ("cpc_remonte01.mf4", lambda p: cpc(p, graine=14, biais_Nm=BIAIS_REMONTAGE_NM)),
        ("cpc_remonte02.mf4", lambda p: cpc(p, graine=15, biais_Nm=BIAIS_REMONTAGE_NM + 1.0)),
    ],
    "5-Décollage en pente": [("pente_01.mf4", lambda p: dynamique(p, graine=21, duree_s=120.0,
                                                                  temperature_fin=32.0))],
    "7-WLTC": [("wltc_01.mf4", lambda p: dynamique(p, graine=31, duree_s=200.0,
                                                   temperature_fin=45.0))],
}


def generer(racine: Path) -> Path:
    racine = Path(racine)
    for dossier, fichiers in ARBORESCENCE.items():
        for nom, fabrique in fichiers:
            fabrique(racine / dossier / nom)
    return racine


if __name__ == "__main__":
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--sortie", default="donnees_synthetiques",
                         help="Dossier racine à créer.")
    args = parseur.parse_args()
    racine = generer(Path(args.sortie))
    print(f"Arborescence synthétique créée dans : {racine.resolve()}")
    print("⚠️ Données FABRIQUÉES — validation du pipeline uniquement.")
