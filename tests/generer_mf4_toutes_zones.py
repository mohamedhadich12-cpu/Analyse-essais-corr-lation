#!/usr/bin/env python3
"""Génère UNE acquisition MDF4 synthétique enchaînant toutes les situations.

⚠️ CES DONNÉES SONT FABRIQUÉES. Elles servent à exercer l'application de bout
en bout — détection des zones, tracés, options d'affichage, analyse — sur un
fichier dont on connaît la réponse exacte. Aucun chiffre qui en sort ne doit
figurer dans le rapport.

À la différence de `generer_mf4_synthetique.py`, qui répartit les situations
sur une arborescence d'essais, celui-ci les rassemble dans **un seul fichier** :

    | Phase                       | Durée  | Ce qu'elle produit                   |
    |-----------------------------|--------|--------------------------------------|
    | A. repos initial            |  20 s  | plage de repos (début) → dérive zéro  |
    | B. balayage montant         |  89 s  | paliers « montée »                    |
    | C. balayage descendant      |  80 s  | paliers « descente » → hystérésis     |
    | D. repos intermédiaire      |  15 s  | plage de repos NON retenue (milieu)   |
    | E. cycle transitoire        | 150 s  | plage dynamique franche → retard      |
    | F. 4 départs arrêtés        | 156 s  | fronts courts → verrou de durée       |
    | G. frémissement de bruit    |  14 s  | plage active ÉCARTÉE (score faible)   |
    | H. repos final              |  22 s  | plage de repos (fin) → dérive zéro    |

    Les bornes exactes, calculées à la construction, figurent dans la fiche de
    vérité terrain écrite à côté du fichier : ce tableau n'en est qu'un aperçu.

Deux grandeurs ne peuvent PAS tenir dans un fichier unique, et ce n'est pas une
limite du générateur mais de la mesure elle-même :

  * la **répétabilité** compare plusieurs exécutions du même essai ;
  * la **répétabilité après remontage** compare deux campagnes séparées par un
    démontage.

Pour celles-là, il faut l'arborescence de `generer_mf4_synthetique.py`.

Le fichier s'accompagne d'un `verite_terrain.md` : les valeurs injectées, à
confronter à ce que l'application affiche.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from asammdf import MDF, Signal

from generer_mf4_synthetique import (  # vérité terrain commune aux deux générateurs
    BRUIT_NM,
    ECART_VOIES_NM,
    FE_COUPLE,
    FE_ETAT,
    FE_TEMPERATURE,
    GAIN,
    HYSTERESIS_NM,
    NOMS,
    OFFSET_NM,
    PLEINE_ECHELLE,
    RETARD_S,
    SENSIBILITE_TH_NM_PAR_C,
)

# --- Vérité terrain propre à ce fichier -------------------------------------
TEMPERATURE_FROIDE = 20.0   # °C — phases statiques, à température stabilisée
TEMPERATURE_CHAUDE = 50.0   # °C — atteinte en fin d'essai
NIVEAUX_BALAYAGE = (0, 150, 300, 450, 600, 750, 900, 1050, 1200)
SOMMET_DA_NM = 1000.0       # crête d'un départ arrêté
RETRO_DA_NM = -150.0        # couple de rétro entre deux départs
# Agitation du couple de rétro (σ, N·m). Au-dessus de la tolérance de
# stabilisation (7,5 N·m) pour qu'aucun palier n'y soit détecté, bien
# au-dessous du seuil d'activité (30 N·m) pour qu'aucune plage
# dynamique n'y naisse : le rétro n'est ni un point de mesure, ni un transitoire.
BRUIT_RETRO_NM = 12.0

# Les phases statiques se font à température stabilisée : c'est ainsi qu'on
# procède au banc, et cela garde le gain et l'offset lisibles sans correction
# thermique. L'excursion thermique commence au cycle transitoire.
DEBUT_ECHAUFFEMENT_S = 215.0


class Profil:
    """Construit un profil de couple de référence phase par phase.

    Le régime est déclaré **explicitement** à chaque phase plutôt que déduit du
    couple : c'est lui qui distingue un repos (arbre à l'arrêt) d'un simple
    passage à couple nul en roulant, et cette distinction ne se devine pas
    depuis le couple seul.
    """

    def __init__(self, fe: float = FE_COUPLE) -> None:
        self.fe = fe
        self._couple: list[np.ndarray] = []
        self._regime: list[np.ndarray] = []
        self.reperes: list[tuple[str, float, float]] = []

    # -- primitives ---------------------------------------------------------
    def _n(self, duree_s: float) -> int:
        return max(2, int(round(duree_s * self.fe)))

    @property
    def duree_s(self) -> float:
        return sum(bloc.size for bloc in self._couple) / self.fe

    def _ajouter(self, couple: np.ndarray, regime: np.ndarray) -> None:
        self._couple.append(couple)
        self._regime.append(regime)

    def phase(self, nom: str):
        """Contexte de nommage : enregistre les bornes de la phase courante."""
        return _Phase(self, nom)

    # -- éléments de profil -------------------------------------------------
    def palier(self, niveau: float, duree_s: float) -> None:
        n = self._n(duree_s)
        self._ajouter(np.full(n, float(niveau)), np.full(n, 50.0 + niveau / 20.0))

    def rampe(self, depart: float, arrivee: float, duree_s: float) -> None:
        n = self._n(duree_s)
        couple = np.linspace(depart, arrivee, n)
        self._ajouter(couple, 50.0 + couple / 20.0)

    def repos(self, duree_s: float) -> None:
        """Couple nul ET arbre à l'arrêt : les deux conditions du relevé de zéro."""
        n = self._n(duree_s)
        self._ajouter(np.zeros(n), np.zeros(n))

    def cycle(self, duree_s: float, graine: int) -> None:
        """Transitoire riche : trois périodes superposées, comme un cycle routier."""
        n = self._n(duree_s)
        u = np.arange(n) / self.fe
        couple = (
            420.0
            + 340.0 * np.sin(2 * np.pi * u / 25.0)
            + 190.0 * np.sin(2 * np.pi * u / 7.3 + 1.1)
            + 90.0 * np.sin(2 * np.pi * u / 3.1 + 0.4)
        )
        self._ajouter(couple, np.abs(couple) / 3.0)

    def retro(self, duree_s: float, graine: int) -> None:
        """Couple de rétro établi, agité comme sur une acquisition réelle.

        Cf. `depart_arrete` pour la raison d'être de cette agitation.
        """
        rng = np.random.default_rng(graine)
        n = self._n(duree_s)
        couple = RETRO_DA_NM + rng.normal(0.0, BRUIT_RETRO_NM, n)
        self._ajouter(couple, np.abs(couple) / 2.5 + 30.0)

    def depart_arrete(self, montee_s: float, descente_s: float, rétro_s: float,
                      graine: int = 0) -> None:
        """Front raide puis retour au couple de rétro, lequel n'est pas lisse.

        La durée d'activité d'un tel front est de quelques secondes seulement —
        moins que la durée minimale d'une fenêtre d'intercorrélation par défaut
        (dix fois le retard maximal recherché). C'est **volontaire** : ce
        fichier doit permettre de constater le verrou, et de le lever avec
        `duree_min_fenetre_s`.

        Le rétro porte une agitation de `BRUIT_RETRO_NM`, comme sur une
        acquisition réelle. Ce n'est pas un ornement : sans elle, ces longues
        sections plates seraient détectées comme un **palier stabilisé** à
        −150 N·m, entreraient dans la régression, et le gain relevé s'écarterait
        de celui qu'on a injecté — alors que rien n'est en cause. L'agitation
        reste bien au-dessous du seuil d'activité, elle ne crée donc aucune
        plage dynamique fantôme.
        """
        rng = np.random.default_rng(graine)
        n_montee = self._n(montee_s)
        montee = np.linspace(RETRO_DA_NM, SOMMET_DA_NM, n_montee)
        n_descente = self._n(descente_s)
        descente = np.linspace(SOMMET_DA_NM, RETRO_DA_NM, n_descente)
        n_retro = self._n(rétro_s)
        retro = RETRO_DA_NM + rng.normal(0.0, BRUIT_RETRO_NM, n_retro)
        for bloc in (montee, descente, retro):
            self._ajouter(bloc, np.abs(bloc) / 2.5 + 30.0)

    def fremissement(self, duree_s: float, amplitude_Nm: float, graine: int) -> None:
        """Agitation qui franchit le seuil d'activité sans rien apprendre.

        Assez remuante pour être détectée comme active, bien trop pauvre pour
        identifier un retard : c'est exactement ce que l'arbitrage par score
        doit écarter, et il faut donc un cas qui l'exerce.
        """
        rng = np.random.default_rng(graine)
        n = self._n(duree_s)
        u = np.arange(n) / self.fe
        couple = 80.0 + amplitude_Nm * np.sin(2 * np.pi * u / 1.7)
        couple += rng.normal(0.0, amplitude_Nm / 4.0, n)
        self._ajouter(couple, np.abs(couple) / 3.0)

    # -- restitution --------------------------------------------------------
    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        couple = np.concatenate(self._couple)
        regime = np.concatenate(self._regime)
        t = np.arange(couple.size) / self.fe
        return t, couple, regime


class _Phase:
    """Enregistre les bornes en secondes d'une phase, pour la fiche de vérité."""

    def __init__(self, profil: Profil, nom: str) -> None:
        self.profil, self.nom = profil, nom

    def __enter__(self):
        self._debut = self.profil.duree_s
        return self.profil

    def __exit__(self, *_):
        self.profil.reperes.append((self.nom, self._debut, self.profil.duree_s))
        return False


def construire_profil(graine: int = 7) -> Profil:
    """Enchaîne les huit phases décrites en tête de module."""
    p = Profil()

    with p.phase("A · repos initial") as ph:
        ph.repos(20.0)

    with p.phase("B · balayage montant") as ph:
        ph.rampe(0.0, NIVEAUX_BALAYAGE[0], 1.0)
        for i, niveau in enumerate(NIVEAUX_BALAYAGE):
            ph.palier(niveau, 8.0)
            if i < len(NIVEAUX_BALAYAGE) - 1:
                ph.rampe(niveau, NIVEAUX_BALAYAGE[i + 1], 2.0)

    with p.phase("C · balayage descendant") as ph:
        descente = NIVEAUX_BALAYAGE[-2::-1]
        for i, niveau in enumerate(descente):
            ph.rampe(NIVEAUX_BALAYAGE[-1] if i == 0 else descente[i - 1], niveau, 2.0)
            ph.palier(niveau, 8.0)

    with p.phase("D · repos intermédiaire") as ph:
        ph.repos(15.0)

    with p.phase("E · cycle transitoire") as ph:
        ph.cycle(150.0, graine=graine)

    with p.phase("F · départs arrêtés") as ph:
        ph.rampe(0.0, RETRO_DA_NM, 2.0)
        # Rétro établi avant le premier front : sans ce répit, l'activité du
        # cycle et celle du front se touchent à moins de `duree_comblement_s`,
        # les deux sont comblées en une seule plage, et le premier front cesse
        # d'être une fenêtre à part entière.
        ph.retro(14.0, graine=graine + 2)
        for i in range(4):
            ph.depart_arrete(montee_s=0.4, descente_s=3.6, rétro_s=31.0,
                             graine=graine + 10 + i)

    with p.phase("G · frémissement écarté") as ph:
        ph.rampe(RETRO_DA_NM, 80.0, 2.0)
        ph.fremissement(12.0, amplitude_Nm=55.0, graine=graine + 1)

    with p.phase("H · repos final") as ph:
        ph.rampe(80.0, 0.0, 2.0)
        ph.repos(20.0)

    return p


def temperature_a(t: np.ndarray) -> np.ndarray:
    """Palier thermique pendant les phases statiques, montée ensuite.

    Un balayage se fait à température stabilisée : y superposer une dérive
    thermique brouillerait le gain et l'offset, qui sont précisément ce que
    cette phase doit permettre de retrouver.
    """
    fin = float(t[-1])
    montee = np.clip((t - DEBUT_ECHAUFFEMENT_S) / max(fin - DEBUT_ECHAUFFEMENT_S, 1.0),
                     0.0, 1.0)
    return TEMPERATURE_FROIDE + (TEMPERATURE_CHAUDE - TEMPERATURE_FROIDE) * montee


def _sens_hysteresis(t: np.ndarray, reperes: list[tuple[str, float, float]]
                     ) -> np.ndarray:
    """−1 sur la phase descendante du balayage, 0 partout ailleurs.

    L'hystérésis est un écart entre un même couple atteint en montant et en
    descendant : elle n'a de sens que sur le balayage, et l'appliquer au cycle
    transitoire fabriquerait un défaut qui n'existe pas.
    """
    descente = next(r for r in reperes if r[0].startswith("C ·"))
    return -1.0 * ((t >= descente[1]) & (t < descente[2]))


def construire_signaux(graine: int = 7):
    """Profil de référence, voies mesurées, régime, température et état."""
    rng = np.random.default_rng(graine)
    profil = construire_profil(graine)
    t, reference, regime = profil.arrays()
    temperature = temperature_a(t)

    # La voie transmissions voit le même couple, retardé, avec erreur de gain,
    # offset, hystérésis sur le balayage et dérive thermique.
    reference_retardee = np.interp(t - RETARD_S, t, reference,
                                   left=reference[0], right=reference[-1])
    mesure = (
        GAIN * reference_retardee
        + OFFSET_NM
        + HYSTERESIS_NM * _sens_hysteresis(t, profil.reperes)
        + SENSIBILITE_TH_NM_PAR_C * (temperature - TEMPERATURE_FROIDE)
    )
    gauche = mesure + ECART_VOIES_NM / 2 + rng.normal(0, BRUIT_NM, t.size)
    droite = mesure - ECART_VOIES_NM / 2 + rng.normal(0, BRUIT_NM, t.size)

    duree = float(t[-1])
    t_temperature = np.arange(0.0, duree, 1.0 / FE_TEMPERATURE)
    temperature_1hz = temperature_a(t_temperature)

    # Canal d'état : une brève perte de liaison télémétrique, pour que la
    # distribution restituée par l'application ne soit pas triviale.
    t_etat = np.arange(0.0, duree, 1.0 / FE_ETAT)
    etat = np.ones(t_etat.size, dtype=np.uint8)
    etat[(t_etat >= 300.0) & (t_etat < 308.0)] = 0

    return profil, t, gauche, droite, reference, regime, (
        t_temperature, temperature_1hz, t_etat, etat)


def ecrire(chemin: Path, graine: int = 7) -> tuple[Path, Profil]:
    """Écrit le MDF4 à trois groupes de cadences distinctes."""
    profil, t, g, d, reference, regime, annexes = construire_signaux(graine)
    t_temperature, temperature, t_etat, etat = annexes

    chemin.parent.mkdir(parents=True, exist_ok=True)
    mdf = MDF(version="4.10")
    mdf.append(
        [
            Signal(g, t, name=NOMS["gauche"], unit="N.m"),
            Signal(d, t, name=NOMS["droite"], unit="N.m"),
            Signal(reference, t, name=NOMS["reference_gauche"], unit="N.m"),
            Signal(reference.copy(), t, name=NOMS["reference_droite"], unit="N.m"),
            Signal(regime, t, name=NOMS["regime"], unit="rpm"),
        ],
        comment="Couples et regime",
    )
    mdf.append([Signal(temperature, t_temperature, name=NOMS["temperature"],
                       unit="degC")], comment="Thermique")
    mdf.append([Signal(etat, t_etat, name=NOMS["etat"], unit="")],
               comment="Etat telemetrie")
    mdf.save(chemin, overwrite=True)
    mdf.close()
    return chemin, profil


def fiche_verite(profil: Profil) -> str:
    """Fiche des valeurs injectées, à confronter à ce qu'affiche l'application."""
    derive_thermique = SENSIBILITE_TH_NM_PAR_C * (
        TEMPERATURE_CHAUDE - TEMPERATURE_FROIDE)
    lignes = [
        "# Vérité terrain — acquisition synthétique « toutes zones »",
        "",
        "⚠️ **Données fabriquées.** Elles servent à vérifier que l'application "
        "retrouve des valeurs connues. Aucun chiffre issu de ce fichier ne doit "
        "figurer dans le rapport.",
        "",
        f"Pleine échelle à déclarer dans l'application : **{PLEINE_ECHELLE:.0f} N·m**.",
        "",
        "## Valeurs injectées",
        "",
        "| Grandeur | Valeur injectée | Où l'application la restitue |",
        "|---|---|---|",
        f"| Erreur de gain | `{GAIN:.4f}`, soit **{100 * (GAIN - 1):+.1f} %** | "
        "régression, pente `a` |",
        f"| Offset | **{OFFSET_NM:+.1f} N·m** "
        "| régression, ordonnée `b` |",
        f"| Hystérésis montée/descente | **{HYSTERESIS_NM:.1f} N·m** "
        "| hystérésis, sur le "
        "balayage (phases B et C) |",
        f"| Retard de la voie transmissions | **{1000 * RETARD_S:+.0f} ms** | "
        "recalage temporel, médiane sur les fenêtres retenues |",
        f"| Sensibilité thermique | **{SENSIBILITE_TH_NM_PAR_C:.2f} N·m/°C** | "
        "sensibilité thermique |",
        f"| Dérive de zéro | **{derive_thermique:+.2f} N·m** "
        "| dérive de zéro |",
        f"| Écart systématique gauche − droite | **{ECART_VOIES_NM:+.1f} N·m** | "
        "redondance G/D, **si la case « essai en ligne droite » est cochée** |",
        f"| Bruit de mesure (σ) | {BRUIT_NM:.1f} N·m | — |",
        "",
        f"La dérive de zéro vient entièrement de l'échauffement : le rotor passe "
        f"de {TEMPERATURE_FROIDE:.0f} à {TEMPERATURE_CHAUDE:.0f} °C entre le "
        f"relevé de zéro initial et le relevé final, soit "
        f"{SENSIBILITE_TH_NM_PAR_C:.2f} × {TEMPERATURE_CHAUDE - TEMPERATURE_FROIDE:.0f} "
        f"= {derive_thermique:.2f} N·m.",
        "",
        "### Deux valeurs qu'il ne faut PAS attendre au chiffre près",
        "",
        "La régression brute mélange trois effets que le rapport sépare ensuite. "
        "Il est normal — et sain — qu'elle ne recrache pas exactement les "
        "constantes injectées :",
        "",
        f"* **l'ordonnée `b` sort un peu au-dessous de {OFFSET_NM:.1f} N·m** "
        "(≈ 5,8). La régression porte sur les paliers des DEUX branches du "
        f"balayage, que l'hystérésis sépare de {HYSTERESIS_NM:.1f} N·m : la "
        "droite passe entre les deux. C'est l'hystérésis, rapportée à part, qui "
        "porte cet écart ;",
        f"* **la pente `a` sort un peu au-dessous de {GAIN:.4f}** (≈ 1,011). Le "
        "relevé de zéro final est pris à chaud : il porte la dérive thermique en "
        "plus de l'offset, et tire l'ajustement vers le haut à couple nul. C'est "
        "la sensibilité thermique, elle aussi rapportée à part, qui porte cet "
        "écart.",
        "",
        "Les autres grandeurs, elles, doivent tomber juste : retard, dérive de "
        "zéro, hystérésis et redondance sont mesurées là où rien ne les brouille.",
        "",
        "## Découpage de l'acquisition",
        "",
        "| Phase | Début (s) | Fin (s) | Ce qui doit être détecté |",
        "|---|---|---|---|",
    ]
    attendu = {
        "A": "plage de repos — celle du **début**, qui sert à la dérive de zéro",
        "B": "neuf paliers, tous marqués « montée »",
        "C": "les mêmes niveaux en « descente » → hystérésis calculable",
        "D": "plage de repos, mais **au milieu** : détectée, non retenue pour la dérive",
        "E": "plage dynamique franche — la mieux notée, celle qui porte le retard",
        "F": "4 fronts courts, séparés par du rétro agité : **invisibles** avec "
             "le réglage par défaut (durée minimale ≈ 5 s), visibles à 2 s",
        "G": "plage active **écartée** : elle franchit le seuil d'activité, son "
             "score reste sous le quart du meilleur",
        "H": "plage de repos — celle de la **fin**, qui ferme la dérive de zéro",
    }
    for nom, debut, fin in profil.reperes:
        lignes.append(f"| {nom} | {debut:.1f} | {fin:.1f} | "
                      f"{attendu.get(nom[0], '—')} |")
    lignes += [
        "",
        "Le rétro des départs arrêtés porte volontairement une agitation de "
        f"σ = {BRUIT_RETRO_NM:.0f} N·m — au-dessus de la tolérance de "
        "stabilisation, au-dessous du seuil d'activité. Sans elle, ces longues "
        "sections plates seraient prises pour un palier stabilisé à "
        f"{RETRO_DA_NM:.0f} N·m et fausseraient la régression.",
        "",
        "## Autres canaux à vérifier",
        "",
        "* **trois cadences distinctes** — couples et régime à "
        f"{FE_COUPLE:.0f} Hz, température à {FE_TEMPERATURE:.0f} Hz, état à "
        f"{FE_ETAT:.0f} Hz : l'onglet 1 doit annoncer trois groupes, et le "
        "rééchantillonnage sur base commune se faire sans extrapolation ;",
        "* **canal d'état** `PCM16_LED_Status` : une coupure de liaison de 8 s "
        "est injectée, la distribution restituée ne doit donc pas être "
        "« 100 % à 1 » ;",
        "* **deux voies de référence** G et D, chargées symétriquement.",
        "",
        "## Ce que ce fichier ne peut pas couvrir",
        "",
        "* la **répétabilité** compare plusieurs exécutions du même essai ;",
        "* la **répétabilité après remontage** compare deux campagnes séparées "
        "par un démontage.",
        "",
        "Ces deux grandeurs demandent plusieurs acquisitions par construction : "
        "l'application doit les déclarer **non calculables**, en le motivant, et "
        "c'est le bon comportement à vérifier ici. Pour les exercer réellement, "
        "utiliser `generer_mf4_synthetique.py`, qui produit l'arborescence "
        "correspondante.",
        "",
        "## Mode d'emploi",
        "",
        "1. onglet **1 · Exploration** : pointer le dossier, *Inventorier les canaux* ;",
        "2. onglet **3 · Canaux** : le mapping se propose seul, les noms sont ceux "
        "des campagnes synthétiques ;",
        f"3. barre latérale : pleine échelle **{PLEINE_ECHELLE:.0f} N·m** ; cocher "
        "**« essai en ligne droite »** pour que la redondance G/D soit calculée ;",
        "4. onglet **4 · Zones détectées** : les phases A à H doivent apparaître "
        "comme annoncé ci-dessus ;",
        "5. onglet **2 · Visualisation** : superposer les familles de zones, et "
        "mesurer le retard au curseur sur un front de départ arrêté ;",
        "6. onglet **5 · Hypothèses** : passer *Durée minimale d'une fenêtre* à "
        "**2 s** pour voir les quatre départs arrêtés devenir des fenêtres "
        "exploitées — le retard passe alors d'une à cinq fenêtres ;",
        "7. onglet **6 · Analyse & résultats** : confronter au tableau des valeurs "
        "injectées.",
        "",
    ]
    return "\n".join(lignes)


def generer(sortie: Path, graine: int = 7) -> Path:
    sortie = Path(sortie)
    chemin, profil = ecrire(sortie / "essai_toutes_zones.mf4", graine)
    (sortie / "verite_terrain.md").write_text(fiche_verite(profil), encoding="utf-8")
    return chemin


if __name__ == "__main__":
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--sortie", default="essai_toutes_zones",
                         help="Dossier à créer, à pointer ensuite depuis l'application.")
    parseur.add_argument("--graine", type=int, default=7,
                         help="Graine du générateur aléatoire (bruit de mesure).")
    args = parseur.parse_args()
    chemin = generer(Path(args.sortie), args.graine)
    print(f"Acquisition synthétique écrite : {chemin.resolve()}")
    print(f"Vérité terrain : {(chemin.parent / 'verite_terrain.md').resolve()}")
    print("⚠️ Données FABRIQUÉES — validation du pipeline uniquement.")
