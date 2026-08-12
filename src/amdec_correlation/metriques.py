"""Grandeurs métrologiques : régression, hystérésis, recalage, incertitude.

Toutes les fonctions de ce module renvoient un objet portant un champ
`non_calculable` : lorsqu'une grandeur ne peut pas être obtenue (canal absent,
excursion thermique insuffisante, pas de relevé de zéro identifiable...), le
motif est renseigné et **aucune valeur n'est produite**. Rien n'est extrapolé.

Conventions :
  * les couples sont en N·m, les températures en °C, les temps en secondes ;
  * « % PE » désigne le pourcentage de la pleine échelle du capteur
    (1500 N·m par défaut, cf. configuration) ;
  * le *résidu* est toujours défini comme `couple_mesuré − couple_référence`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import signal as sig_scipy
from scipy import stats

from .config import (
    ParamsDiagnostic,
    ParamsIntercorrelation,
    ParamsPaliers,
    ParamsThermique,
    ParamsZero,
)

# ---------------------------------------------------------------------------
# Utilitaires numériques
# ---------------------------------------------------------------------------


def frequence_echantillonnage(t: np.ndarray) -> float:
    """Fréquence de la grille de temps (supposée uniforme après rééchantillonnage)."""
    if t.size < 2:
        return float("nan")
    return float((t.size - 1) / (t[-1] - t[0]))


def _std_glissant(x: np.ndarray, n: int) -> np.ndarray:
    """Écart-type glissant (fenêtre centrée de n points), en O(N).

    Le signal est recentré avant le cumsum : sans cela, la différence de sommes
    de carrés sur un signal fortement décalé perd sa précision numérique.
    Les bords, où la fenêtre est incomplète, valent `inf` — ils ne peuvent donc
    jamais être déclarés stabilisés.
    """
    x = np.asarray(x, dtype=float)
    sortie = np.full(x.size, np.inf)
    if n < 2 or x.size < n:
        return sortie
    xc = x - float(np.nanmean(x))
    c1 = np.concatenate(([0.0], np.cumsum(xc)))
    c2 = np.concatenate(([0.0], np.cumsum(xc * xc)))
    s1 = c1[n:] - c1[:-n]
    s2 = c2[n:] - c2[:-n]
    var = np.maximum((s2 - s1 * s1 / n) / (n - 1), 0.0)
    demi = n // 2
    sortie[demi : demi + var.size] = np.sqrt(var)
    return sortie


def _segments(masque: np.ndarray, longueur_min: int) -> list[tuple[int, int]]:
    """Plages contiguës [début, fin) où `masque` est vrai, d'au moins `longueur_min` points."""
    if masque.size == 0:
        return []
    m = np.asarray(masque, dtype=bool).astype(np.int8)
    bords = np.diff(np.concatenate(([0], m, [0])))
    debuts = np.flatnonzero(bords == 1)
    fins = np.flatnonzero(bords == -1)
    return [(int(d), int(f)) for d, f in zip(debuts, fins) if f - d >= longueur_min]


def _combler_trous(masque: np.ndarray, longueur_max: int) -> np.ndarray:
    """Comble les interruptions internes courtes d'un masque booléen.

    Fermeture morphologique : les plages fausses de moins de `longueur_max`
    points, encadrées de part et d'autre par des plages vraies, sont passées à
    vrai. Les bords ne sont jamais comblés.
    """
    m = np.asarray(masque, dtype=bool).copy()
    if longueur_max <= 0:
        return m
    for debut, fin in _segments(~m, 1):
        if debut > 0 and fin < m.size and (fin - debut) <= longueur_max:
            m[debut:fin] = True
    return m


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x * x))) if x.size else float("nan")


# ---------------------------------------------------------------------------
# Régression linéaire
# ---------------------------------------------------------------------------


@dataclass
class Regression:
    """Régression `y = a·x + b` par moindres carrés ordinaires."""

    a: float
    b: float
    r2: float
    sigma_a: float
    sigma_b: float
    p_value: float
    n: int
    x: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)
    residus: np.ndarray = field(repr=False)
    non_calculable: str | None = None

    @property
    def residu_max(self) -> float:
        return float(np.max(np.abs(self.residus))) if self.residus.size else float("nan")

    @property
    def ecart_type_residus(self) -> float:
        if self.n <= 2:
            return float("nan")
        return float(np.sqrt(np.sum(self.residus**2) / (self.n - 2)))


def regression(x: Sequence[float], y: Sequence[float]) -> Regression:
    """Régression linéaire avec incertitudes-types sur la pente et l'ordonnée."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    vide = np.array([])
    if x.size < 3:
        return Regression(
            float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
            float("nan"), int(x.size), x, y, vide,
            non_calculable=f"régression impossible : {x.size} point(s) exploitable(s), 3 minimum requis",
        )
    if np.ptp(x) == 0:
        return Regression(
            float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
            float("nan"), int(x.size), x, y, vide,
            non_calculable="régression impossible : la variable explicative est constante",
        )

    res = stats.linregress(x, y)
    residus = y - (res.slope * x + res.intercept)
    return Regression(
        a=float(res.slope),
        b=float(res.intercept),
        r2=float(res.rvalue**2),
        sigma_a=float(res.stderr),
        sigma_b=float(res.intercept_stderr),
        p_value=float(res.pvalue),
        n=int(x.size),
        x=x,
        y=y,
        residus=residus,
    )


# ---------------------------------------------------------------------------
# Paliers stabilisés
# ---------------------------------------------------------------------------


@dataclass
class Palier:
    """Un point stabilisé : moyennes sur la partie finale de la plage stable."""

    t_debut: float
    t_fin: float
    reference: float
    mesure: float
    ecart_type_mesure: float
    ecart_type_reference: float
    n: int
    sens: str = "indetermine"  # "montee" | "descente"
    temperature: float | None = None
    source: str = ""

    @property
    def residu(self) -> float:
        return self.mesure - self.reference


def detecter_paliers(
    t: np.ndarray,
    reference: np.ndarray,
    mesure: np.ndarray,
    pleine_echelle_Nm: float,
    params: ParamsPaliers,
    temperature: np.ndarray | None = None,
    source: str = "",
) -> list[Palier]:
    """Segmente un essai en points stabilisés.

    HYPOTHÈSES (à citer telles quelles dans le rapport) :
      1. Le critère de stabilisation porte sur le **couple de référence** : une
         fenêtre glissante de `duree_palier_s` est déclarée stable si l'écart-type
         du couple de référence y reste sous `tolerance_stab_pc_pe` % PE. Le bruit
         de la voie mesurée n'intervient pas dans le critère, sans quoi une voie
         bruitée invaliderait des paliers pourtant établis côté banc.
      2. La valeur retenue est la moyenne sur la **fraction finale**
         (`fraction_finale`) de la plage stable, pour écarter le transitoire
         d'établissement thermique et mécanique.
      3. Deux plages successives dont les niveaux de référence diffèrent de moins
         de `ecart_min_paliers_pc_pe` % PE sont fusionnées : il s'agit du même
         palier, momentanément interrompu par une perturbation.
      4. Le sens (montée / descente) est déduit du signe de la variation de
         niveau par rapport au palier précédent ; le premier palier hérite du
         sens du second.
    """
    t = np.asarray(t, dtype=float)
    fe = frequence_echantillonnage(t)
    n_fenetre = max(3, int(round(params.duree_palier_s * fe)))
    tolerance_Nm = params.tolerance_stab_pc_pe * pleine_echelle_Nm / 100.0

    stable = _std_glissant(reference, n_fenetre) <= tolerance_Nm
    plages = _segments(stable, n_fenetre)

    bruts: list[Palier] = []
    for i0, i1 in plages:
        # Hypothèse 2 : on ne garde que la fraction finale de la plage.
        debut = i1 - max(2, int(round((i1 - i0) * params.fraction_finale)))
        tranche = slice(debut, i1)
        bruts.append(
            Palier(
                t_debut=float(t[debut]),
                t_fin=float(t[i1 - 1]),
                reference=float(np.mean(reference[tranche])),
                mesure=float(np.mean(mesure[tranche])),
                ecart_type_mesure=float(np.std(mesure[tranche], ddof=1)),
                ecart_type_reference=float(np.std(reference[tranche], ddof=1)),
                n=int(i1 - debut),
                temperature=(float(np.mean(temperature[tranche])) if temperature is not None else None),
                source=source,
            )
        )

    paliers = _fusionner_paliers(bruts, params, pleine_echelle_Nm)
    _attribuer_sens(paliers, params, pleine_echelle_Nm)
    return paliers


def _fusionner_paliers(
    paliers: list[Palier], params: ParamsPaliers, pe: float
) -> list[Palier]:
    """Hypothèse 3 : fusionne les plages successives de même niveau de référence."""
    if not paliers:
        return []
    ecart_min = params.ecart_min_paliers_pc_pe * pe / 100.0
    fusionnes = [paliers[0]]
    for p in paliers[1:]:
        precedent = fusionnes[-1]
        if abs(p.reference - precedent.reference) < ecart_min:
            poids_p, poids_prec = p.n, precedent.n
            total = poids_p + poids_prec
            fusionnes[-1] = Palier(
                t_debut=precedent.t_debut,
                t_fin=p.t_fin,
                reference=(precedent.reference * poids_prec + p.reference * poids_p) / total,
                mesure=(precedent.mesure * poids_prec + p.mesure * poids_p) / total,
                ecart_type_mesure=max(precedent.ecart_type_mesure, p.ecart_type_mesure),
                ecart_type_reference=max(precedent.ecart_type_reference, p.ecart_type_reference),
                n=total,
                temperature=(
                    None
                    if precedent.temperature is None or p.temperature is None
                    else (precedent.temperature * poids_prec + p.temperature * poids_p) / total
                ),
                source=precedent.source,
            )
        else:
            fusionnes.append(p)
    return fusionnes


def _attribuer_sens(paliers: list[Palier], params: ParamsPaliers, pe: float) -> None:
    """Hypothèse 4 : sens déduit de la variation de niveau au palier précédent."""
    if len(paliers) < 2:
        return
    seuil = params.ecart_min_paliers_pc_pe * pe / 100.0
    for i in range(1, len(paliers)):
        delta = paliers[i].reference - paliers[i - 1].reference
        if abs(delta) < seuil:
            paliers[i].sens = paliers[i - 1].sens
        else:
            paliers[i].sens = "montee" if delta > 0 else "descente"
    paliers[0].sens = paliers[1].sens


# ---------------------------------------------------------------------------
# Hystérésis / non-linéarité
# ---------------------------------------------------------------------------


@dataclass
class Hysteresis:
    max_pc_pe: float = float("nan")
    moyenne_pc_pe: float = float("nan")
    n_appariements: int = 0
    appariements: list[tuple[Palier, Palier, float]] = field(default_factory=list, repr=False)
    non_calculable: str | None = None


def hysteresis(
    paliers: Sequence[Palier], pleine_echelle_Nm: float, params: ParamsPaliers
) -> Hysteresis:
    """Écart montée/descente au même couple de référence, en % PE.

    HYPOTHÈSE : un palier de montée et un palier de descente sont appariés si
    leurs couples de référence diffèrent de moins de
    `tolerance_appariement_pc_pe` % PE ; en cas de candidats multiples, le plus
    proche en couple de référence est retenu.
    """
    montees = [p for p in paliers if p.sens == "montee"]
    descentes = [p for p in paliers if p.sens == "descente"]
    if not montees or not descentes:
        return Hysteresis(
            non_calculable=(
                "hystérésis non calculable : le balayage ne comporte pas à la fois "
                f"une phase montante ({len(montees)} palier(s)) et une phase "
                f"descendante ({len(descentes)} palier(s))"
            )
        )

    tolerance = params.tolerance_appariement_pc_pe * pleine_echelle_Nm / 100.0
    appariements: list[tuple[Palier, Palier, float]] = []
    for m in montees:
        candidats = [d for d in descentes if abs(d.reference - m.reference) <= tolerance]
        if not candidats:
            continue
        d = min(candidats, key=lambda c: abs(c.reference - m.reference))
        appariements.append((m, d, m.mesure - d.mesure))

    if not appariements:
        return Hysteresis(
            non_calculable=(
                "hystérésis non calculable : aucun palier de montée n'a de palier de "
                f"descente au même couple de référence (tolérance "
                f"{params.tolerance_appariement_pc_pe} % PE)"
            )
        )

    ecarts = np.array([abs(e) for _, _, e in appariements])
    return Hysteresis(
        max_pc_pe=float(100.0 * ecarts.max() / pleine_echelle_Nm),
        moyenne_pc_pe=float(100.0 * ecarts.mean() / pleine_echelle_Nm),
        n_appariements=len(appariements),
        appariements=appariements,
    )


def non_linearite_pc_pe(reg: Regression, pleine_echelle_Nm: float) -> float:
    """Résidu maximal par rapport à la droite de régression, en % PE."""
    if reg.non_calculable or not reg.residus.size:
        return float("nan")
    return float(100.0 * reg.residu_max / pleine_echelle_Nm)


# ---------------------------------------------------------------------------
# Répétabilité
# ---------------------------------------------------------------------------


@dataclass
class GroupeRepetabilite:
    reference_moyenne: float
    n: int
    ecart_type_residu_Nm: float
    etendue_Nm: float
    sources: list[str] = field(default_factory=list)


@dataclass
class Repetabilite:
    ecart_type_pc_pe: float = float("nan")
    ecart_type_Nm: float = float("nan")
    degres_liberte: int = 0
    n_points: int = 0
    groupes: list[GroupeRepetabilite] = field(default_factory=list, repr=False)
    non_calculable: str | None = None


def repetabilite(
    paliers: Sequence[Palier], pleine_echelle_Nm: float, params: ParamsPaliers
) -> Repetabilite:
    """Dispersion de la mesure à couple de référence constant.

    HYPOTHÈSES :
      * les paliers sont regroupés par niveau de couple de référence, avec une
        tolérance de `tolerance_appariement_pc_pe` % PE ;
      * la dispersion est calculée sur le **résidu** (mesuré − référence) et non
        sur la mesure brute : à l'intérieur d'un groupe le couple de référence
        n'est pas rigoureusement identique d'une répétition à l'autre, et
        raisonner sur le résidu retire cette variation-là de la dispersion
        attribuée à la chaîne de mesure ;
      * les écarts-types des groupes sont combinés en écart-type *poolé* (somme
        pondérée des variances par les degrés de liberté), ce qui suppose une
        dispersion homogène sur la plage de couple.
    """
    exploitables = [p for p in paliers if math.isfinite(p.residu)]
    if len(exploitables) < 2:
        return Repetabilite(
            non_calculable=(
                f"répétabilité non calculable : {len(exploitables)} palier(s) exploitable(s), "
                "au moins 2 répétitions au même couple de référence sont nécessaires"
            )
        )

    tolerance = params.tolerance_appariement_pc_pe * pleine_echelle_Nm / 100.0
    tries = sorted(exploitables, key=lambda p: p.reference)
    groupes_bruts: list[list[Palier]] = [[tries[0]]]
    for p in tries[1:]:
        if abs(p.reference - groupes_bruts[-1][0].reference) <= tolerance:
            groupes_bruts[-1].append(p)
        else:
            groupes_bruts.append([p])

    groupes: list[GroupeRepetabilite] = []
    somme_var, somme_ddl = 0.0, 0
    for g in groupes_bruts:
        if len(g) < 2:
            continue
        residus = np.array([p.residu for p in g])
        s = float(np.std(residus, ddof=1))
        groupes.append(
            GroupeRepetabilite(
                reference_moyenne=float(np.mean([p.reference for p in g])),
                n=len(g),
                ecart_type_residu_Nm=s,
                etendue_Nm=float(np.ptp(residus)),
                sources=sorted({p.source for p in g if p.source}),
            )
        )
        somme_var += (len(g) - 1) * s * s
        somme_ddl += len(g) - 1

    if somme_ddl == 0:
        return Repetabilite(
            non_calculable=(
                "répétabilité non calculable : aucun niveau de couple de référence n'est "
                f"répété au moins 2 fois (tolérance de regroupement "
                f"{params.tolerance_appariement_pc_pe} % PE)"
            )
        )

    s_poole = math.sqrt(somme_var / somme_ddl)
    return Repetabilite(
        ecart_type_pc_pe=100.0 * s_poole / pleine_echelle_Nm,
        ecart_type_Nm=s_poole,
        degres_liberte=somme_ddl,
        n_points=sum(g.n for g in groupes),
        groupes=groupes,
    )


# ---------------------------------------------------------------------------
# Recalage temporel par intercorrélation
# ---------------------------------------------------------------------------


@dataclass
class Recalage:
    retard_ms: float = float("nan")
    correlation_pic: float = float("nan")
    rms_residu_avant_Nm: float = float("nan")
    rms_residu_apres_Nm: float = float("nan")
    fenetre: tuple[float, float] = (float("nan"), float("nan"))
    duree_fenetre_s: float = float("nan")
    retards_ms: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    correlation: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    non_calculable: str | None = None

    @property
    def gain_rms(self) -> float:
        """Réduction relative du RMS de résidu apportée par le recalage."""
        if not math.isfinite(self.rms_residu_avant_Nm) or self.rms_residu_avant_Nm == 0:
            return float("nan")
        return 1.0 - self.rms_residu_apres_Nm / self.rms_residu_avant_Nm


def plage_dynamique(
    t: np.ndarray,
    reference: np.ndarray,
    pleine_echelle_Nm: float,
    params: ParamsIntercorrelation,
) -> tuple[int, int] | None:
    """La plus longue plage contiguë où le signal est réellement dynamique.

    Critère : écart-type glissant du couple de référence au-dessus du seuil
    d'activité. Les interruptions courtes sont comblées — un cycle transitoire
    passe par des extrema où la variance instantanée s'annule sans cesser
    d'être dynamique.

    Renvoie `None` si aucune plage n'atteint la durée minimale, fixée à dix
    fois le retard maximal recherché : en deçà, le pic d'intercorrélation n'est
    pas identifiable.
    """
    fe = frequence_echantillonnage(t)
    retard_max_s = params.retard_max_ms / 1000.0
    n_activite = max(3, int(round(params.fenetre_activite_s * fe)))
    seuil_Nm = params.seuil_activite_pc_pe * pleine_echelle_Nm / 100.0

    activite = _std_glissant(reference, n_activite)
    actif = np.isfinite(activite) & (activite >= seuil_Nm)
    actif = _combler_trous(actif, int(round(params.duree_comblement_s * fe)))
    duree_min = max(int(round(10 * retard_max_s * fe)), int(round(2.0 * fe)))
    plages = _segments(actif, duree_min)
    if not plages:
        return None
    return max(plages, key=lambda p: p[1] - p[0])


def plages_de_repos(
    t: np.ndarray,
    reference: np.ndarray,
    regime: np.ndarray | None,
    pleine_echelle_Nm: float,
    params: ParamsZero,
) -> list[tuple[int, int]]:
    """Plages de repos : couple de référence et régime sous leurs seuils.

    Ce sont les candidates aux relevés de zéro. La position dans l'essai n'est
    pas jugée ici — c'est `derive_zero` qui exige d'en trouver une au début et
    une à la fin.
    """
    fe = frequence_echantillonnage(t)
    n_min = max(3, int(round(params.duree_fenetre_s * fe)))
    seuil_Nm = params.seuil_couple_ref_pc_pe * pleine_echelle_Nm / 100.0

    masque = np.abs(reference) < seuil_Nm
    if regime is not None and params.seuil_regime is not None:
        masque = masque & (np.abs(regime) < params.seuil_regime)
    return _segments(masque, n_min)


def appliquer_retard(t: np.ndarray, y: np.ndarray, retard_s: float) -> np.ndarray:
    """Avance le signal `y` de `retard_s` secondes (compense un retard positif).

    Convention : un retard positif signifie que `y` est **en retard** sur la
    référence ; on l'avance donc en l'échantillonnant en `t + retard_s`.
    """
    return np.interp(t + retard_s, t, y, left=np.nan, right=np.nan)


def recalage_temporel(
    t: np.ndarray,
    reference: np.ndarray,
    mesure: np.ndarray,
    pleine_echelle_Nm: float,
    params: ParamsIntercorrelation,
) -> Recalage:
    """Retard de la voie mesurée sur la référence, par intercorrélation.

    HYPOTHÈSES :
      1. Seule la plage contiguë la plus longue où le signal est réellement
         *dynamique* est corrélée : l'écart-type glissant du couple de référence
         sur `fenetre_activite_s` doit y dépasser `seuil_activite_pc_pe` % PE.
         Sur une portion quasi stationnaire, le pic d'intercorrélation est plat
         et le retard n'est pas identifiable. Les interruptions de moins de
         `duree_comblement_s` sont comblées au préalable : un cycle transitoire
         passe par des extrema où la variance instantanée s'annule sans cesser
         d'être dynamique, et la fenêtre d'analyse serait sinon fragmentée en
         tronçons trop courts.
      2. Les deux signaux sont filtrés passe-haut à `passe_haut_Hz`
         (Butterworth ordre 2, phase nulle par `filtfilt`) puis centrés-réduits.
         Sans ce filtrage, l'offset et les composantes lentes dominent le produit
         de corrélation et masquent le pic. La phase nulle est indispensable :
         un filtre à phase non nulle introduirait lui-même un retard.
      3. La recherche est bornée à ±`retard_max_ms`.
      4. Le maximum est affiné au sous-échantillon par interpolation parabolique
         sur les trois points entourant le pic — la résolution du retard n'est
         donc pas limitée à la période d'échantillonnage.

    Convention de signe : retard > 0 ⇒ la voie mesurée est **en retard** sur la
    référence banc.
    """
    t = np.asarray(t, dtype=float)
    reference = np.asarray(reference, dtype=float)
    mesure = np.asarray(mesure, dtype=float)
    fe = frequence_echantillonnage(t)
    retard_max_s = params.retard_max_ms / 1000.0

    # Hypothèse 1 : sélection de la plage dynamique la plus longue.
    duree_min_points = max(int(round(10 * retard_max_s * fe)), int(round(2.0 * fe)))
    plage = plage_dynamique(t, reference, pleine_echelle_Nm, params)
    if plage is None:
        return Recalage(
            non_calculable=(
                "recalage non calculable : aucune plage dynamique contiguë d'au moins "
                f"{duree_min_points / fe:.1f} s avec un écart-type de couple de référence "
                f"supérieur à {params.seuil_activite_pc_pe} % PE"
            )
        )
    i0, i1 = plage
    tr = slice(i0, i1)
    x, y = reference[tr].copy(), mesure[tr].copy()

    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < duree_min_points:
        return Recalage(non_calculable="recalage non calculable : trop peu d'échantillons valides")

    rms_avant = rms(mesure[tr] - reference[tr])

    # Hypothèse 2 : passe-haut à phase nulle, puis centrage-réduction.
    if params.passe_haut_Hz:
        wn = params.passe_haut_Hz / (fe / 2.0)
        if 0 < wn < 1:
            b, a = sig_scipy.butter(2, wn, btype="highpass")
            x = sig_scipy.filtfilt(b, a, x)
            y = sig_scipy.filtfilt(b, a, y)
    x = x - x.mean()
    y = y - y.mean()
    if x.std() == 0 or y.std() == 0:
        return Recalage(non_calculable="recalage non calculable : un des signaux est constant après filtrage")
    x /= x.std()
    y /= y.std()

    correlation = sig_scipy.correlate(y, x, mode="full")
    decalages = sig_scipy.correlation_lags(y.size, x.size, mode="full")
    correlation = correlation / x.size  # coefficient de corrélation normalisé

    # Hypothèse 3 : bornage de la recherche.
    dans_bornes = np.abs(decalages) <= max(1, int(round(retard_max_s * fe)))
    if not dans_bornes.any():
        return Recalage(non_calculable="recalage non calculable : fenêtre de recherche vide")
    idx_locaux = np.flatnonzero(dans_bornes)
    k = idx_locaux[int(np.argmax(correlation[dans_bornes]))]

    # Hypothèse 4 : affinage parabolique au sous-échantillon.
    delta = 0.0
    if 0 < k < correlation.size - 1:
        c0, c1, c2 = correlation[k - 1], correlation[k], correlation[k + 1]
        denominateur = c0 - 2 * c1 + c2
        if denominateur != 0:
            delta = float(np.clip(0.5 * (c0 - c2) / denominateur, -1.0, 1.0))

    retard_s = (decalages[k] + delta) / fe
    mesure_recalee = appliquer_retard(t, mesure, retard_s)
    valides = np.isfinite(mesure_recalee[tr])
    rms_apres = rms((mesure_recalee[tr] - reference[tr])[valides])

    return Recalage(
        retard_ms=float(retard_s * 1000.0),
        correlation_pic=float(correlation[k]),
        rms_residu_avant_Nm=rms_avant,
        rms_residu_apres_Nm=rms_apres,
        fenetre=(float(t[i0]), float(t[i1 - 1])),
        duree_fenetre_s=float(t[i1 - 1] - t[i0]),
        retards_ms=decalages[dans_bornes] / fe * 1000.0,
        correlation=correlation[dans_bornes],
    )


# ---------------------------------------------------------------------------
# Sensibilité thermique
# ---------------------------------------------------------------------------


def _moindres_carres_multiple(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """OLS multiple avec incertitudes-types des coefficients et R².

    `X` inclut déjà sa colonne de constante. Renvoie (coefficients, écarts-types,
    R²). Sert à séparer l'effet du couple de l'effet de la température.
    """
    coefficients, *_ = np.linalg.lstsq(X, y, rcond=None)
    residus = y - X @ coefficients
    ddl = X.shape[0] - X.shape[1]
    if ddl <= 0:
        return coefficients, np.full(coefficients.shape, np.nan), float("nan")
    variance = float(residus @ residus) / ddl
    try:
        covariance = variance * np.linalg.inv(X.T @ X)
        erreurs = np.sqrt(np.diag(covariance))
    except np.linalg.LinAlgError:
        erreurs = np.full(coefficients.shape, np.nan)
    total = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(residus @ residus) / total if total > 0 else float("nan")
    return coefficients, erreurs, r2


@dataclass
class SensibiliteThermique:
    pente_Nm_par_C: float = float("nan")
    # Coefficient du couple dans la régression multiple (≈ erreur de gain).
    # NaN si aucune correction du couple n'a été appliquée.
    coefficient_couple: float = float("nan")
    correction_couple: bool = False
    pc_pe_par_C: float = float("nan")
    pc_pe_pour_10C: float = float("nan")
    incertitude_pente_Nm_par_C: float = float("nan")
    r2: float = float("nan")
    p_value: float = float("nan")
    amplitude_C: float = float("nan")
    n_classes: int = 0
    temperatures: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    residus: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    nuage_T: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    nuage_res: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    non_calculable: str | None = None


def sensibilite_thermique(
    temperature: np.ndarray,
    residu: np.ndarray,
    pleine_echelle_Nm: float,
    params: ParamsThermique,
    n_classes: int = 20,
    agreger: bool = True,
    couple: np.ndarray | None = None,
    n_classes_couple: int = 8,
) -> SensibiliteThermique:
    """Régression du résidu sur la température.

    HYPOTHÈSES :
      * en deçà de `amplitude_min_C` d'excursion thermique, la pente n'est pas
        identifiable : la grandeur est déclarée non calculable, pas extrapolée ;
      * la régression est faite sur des **moyennes par classe de température**
        (`n_classes` classes de largeur égale) et non sur les échantillons bruts.
        Un signal à 100 Hz est fortement autocorrélé : régresser les échantillons
        bruts donnerait un nombre de degrés de liberté fictif et une incertitude
        de pente dramatiquement sous-estimée. Les classes vides sont ignorées.
        `agreger=False` court-circuite cette agrégation lorsque les points sont
        déjà des moyennes indépendantes (un point par palier stabilisé).
      * si le couple de référence est fourni, la régression est **multiple** :
        `résidu = α·couple + β·température + γ`, et la sensibilité thermique
        retenue est β. C'est indispensable dès qu'il existe une erreur de gain :
        le résidu croît alors avec le couple, et comme couple et température
        sont corrélés au fil d'un essai (l'échauffement suit la sollicitation),
        une régression sur la seule température attribuerait à l'effet
        thermique une part de l'erreur de gain. L'agrégation se fait alors sur
        une grille croisée température × couple, ce qui décorrèle les deux
        prédicteurs avant l'ajustement.
    """
    T = np.asarray(temperature, dtype=float)
    r = np.asarray(residu, dtype=float)
    C = np.asarray(couple, dtype=float) if couple is not None else None
    ok = np.isfinite(T) & np.isfinite(r)
    if C is not None:
        ok &= np.isfinite(C)
        C = C[ok]
    T, r = T[ok], r[ok]
    if T.size < 3:
        return SensibiliteThermique(
            non_calculable="sensibilité thermique non calculable : pas de données température exploitables"
        )

    amplitude = float(np.ptp(T))
    if amplitude < params.amplitude_min_C:
        return SensibiliteThermique(
            amplitude_C=amplitude,
            nuage_T=T,
            nuage_res=r,
            non_calculable=(
                f"sensibilité thermique non calculable : excursion thermique de "
                f"{amplitude:.1f} °C sur l'essai, inférieure au minimum de "
                f"{params.amplitude_min_C:.1f} °C requis pour identifier une pente"
            ),
        )

    def _classes(valeurs: np.ndarray, nombre: int) -> np.ndarray:
        bords = np.linspace(valeurs.min(), valeurs.max(), nombre + 1)
        return np.clip(np.digitize(valeurs, bords) - 1, 0, nombre - 1)

    # -- agrégation ---------------------------------------------------------
    T_classes: list[float] = []
    r_classes: list[float] = []
    C_classes: list[float] = []
    if not agreger:
        T_classes = list(map(float, T))
        r_classes = list(map(float, r))
        C_classes = list(map(float, C)) if C is not None else []
    elif C is None:
        indices = _classes(T, n_classes)
        for k in range(n_classes):
            m = indices == k
            if m.sum() >= 2:
                T_classes.append(float(T[m].mean()))
                r_classes.append(float(r[m].mean()))
    else:
        # Grille croisée température × couple : décorrèle les deux prédicteurs
        # avant l'ajustement, de sorte que la pente en température ne capte pas
        # l'erreur de gain.
        i_T, i_C = _classes(T, n_classes), _classes(C, n_classes_couple)
        for kt in range(n_classes):
            for kc in range(n_classes_couple):
                m = (i_T == kt) & (i_C == kc)
                if m.sum() >= 2:
                    T_classes.append(float(T[m].mean()))
                    r_classes.append(float(r[m].mean()))
                    C_classes.append(float(C[m].mean()))

    minimum = 4 if C is not None else 3
    if len(T_classes) < minimum:
        return SensibiliteThermique(
            amplitude_C=amplitude,
            nuage_T=T,
            nuage_res=r,
            non_calculable=(
                f"sensibilité thermique non calculable : seulement {len(T_classes)} cellule(s) "
                f"peuplée(s) dans la grille d'agrégation, {minimum} minimum requises"
            ),
        )

    # -- ajustement ---------------------------------------------------------
    T_arr = np.asarray(T_classes)
    r_arr = np.asarray(r_classes)
    if C is None:
        reg = regression(T_classes, r_classes)
        if reg.non_calculable:
            return SensibiliteThermique(
                amplitude_C=amplitude, nuage_T=T, nuage_res=r, non_calculable=reg.non_calculable
            )
        pente, erreur_pente, r2 = reg.a, reg.sigma_a, reg.r2
        p_value, alpha = reg.p_value, float("nan")
        nuage_res, residus_affiches = r, r_arr
    else:
        C_arr = np.asarray(C_classes)
        if np.ptp(T_arr) == 0 or np.ptp(C_arr) == 0:
            return SensibiliteThermique(
                amplitude_C=amplitude, nuage_T=T, nuage_res=r,
                non_calculable="sensibilité thermique non calculable : couple ou température "
                "constant après agrégation",
            )
        X = np.column_stack([C_arr, T_arr, np.ones_like(T_arr)])
        coefficients, erreurs, r2 = _moindres_carres_multiple(X, r_arr)
        alpha, pente = float(coefficients[0]), float(coefficients[1])
        erreur_pente, p_value = float(erreurs[1]), float("nan")
        # Pour l'affichage, on retire la part attribuée au couple : le nuage
        # tracé correspond alors à la pente effectivement ajustée.
        nuage_res = r - alpha * C
        residus_affiches = r_arr - alpha * C_arr

    return SensibiliteThermique(
        pente_Nm_par_C=pente,
        coefficient_couple=alpha,
        correction_couple=C is not None,
        pc_pe_par_C=100.0 * pente / pleine_echelle_Nm,
        pc_pe_pour_10C=100.0 * pente * 10.0 / pleine_echelle_Nm,
        incertitude_pente_Nm_par_C=erreur_pente,
        r2=r2,
        p_value=p_value,
        amplitude_C=amplitude,
        n_classes=len(T_classes),
        temperatures=T_arr,
        residus=residus_affiches,
        nuage_T=T,
        nuage_res=nuage_res,
    )


# ---------------------------------------------------------------------------
# Dérive du zéro
# ---------------------------------------------------------------------------


@dataclass
class DeriveZero:
    zero_debut_Nm: float = float("nan")
    zero_fin_Nm: float = float("nan")
    derive_Nm: float = float("nan")
    derive_pc_pe: float = float("nan")
    fenetre_debut: tuple[float, float] = (float("nan"), float("nan"))
    fenetre_fin: tuple[float, float] = (float("nan"), float("nan"))
    non_calculable: str | None = None


def derive_zero(
    t: np.ndarray,
    mesure: np.ndarray,
    reference: np.ndarray,
    regime: np.ndarray | None,
    pleine_echelle_Nm: float,
    params: ParamsZero,
) -> DeriveZero:
    """Dérive du zéro entre le début et la fin de l'essai.

    HYPOTHÈSES :
      * un relevé de zéro est une plage contiguë d'au moins `duree_fenetre_s`
        pendant laquelle |couple de référence| < `seuil_couple_ref_pc_pe` % PE
        et, si le canal existe, |régime| < `seuil_regime` ;
      * la plage « début » doit commencer dans la première `fraction_bord` de
        l'essai, la plage « fin » se terminer dans la dernière `fraction_bord` ;
        sans quoi il ne s'agit pas d'un relevé de zéro avant/après mais d'un
        simple passage à couple nul en cours d'essai, et la grandeur est
        déclarée non calculable.
    """
    t = np.asarray(t, dtype=float)
    plages = plages_de_repos(t, reference, regime, pleine_echelle_Nm, params)
    if len(plages) < 2:
        return DeriveZero(
            non_calculable=(
                f"dérive de zéro non calculable : {len(plages)} plage(s) de repos d'au moins "
                f"{params.duree_fenetre_s:.0f} s identifiée(s) (il en faut une en début et une "
                "en fin d'essai)"
            )
        )

    t0, t1 = float(t[0]), float(t[-1])
    duree = t1 - t0
    bord = params.fraction_bord * duree
    premiere, derniere = plages[0], plages[-1]
    if t[premiere[0]] > t0 + bord:
        return DeriveZero(
            non_calculable=(
                "dérive de zéro non calculable : aucun relevé de zéro dans les premiers "
                f"{params.fraction_bord * 100:.0f} % de l'essai"
            )
        )
    if t[derniere[1] - 1] < t1 - bord:
        return DeriveZero(
            non_calculable=(
                "dérive de zéro non calculable : aucun relevé de zéro dans les derniers "
                f"{params.fraction_bord * 100:.0f} % de l'essai"
            )
        )

    zero_debut = float(np.mean(mesure[premiere[0] : premiere[1]]))
    zero_fin = float(np.mean(mesure[derniere[0] : derniere[1]]))
    derive = zero_fin - zero_debut
    return DeriveZero(
        zero_debut_Nm=zero_debut,
        zero_fin_Nm=zero_fin,
        derive_Nm=derive,
        derive_pc_pe=100.0 * derive / pleine_echelle_Nm,
        fenetre_debut=(float(t[premiere[0]]), float(t[premiere[1] - 1])),
        fenetre_fin=(float(t[derniere[0]]), float(t[derniere[1] - 1])),
    )


# ---------------------------------------------------------------------------
# Redondance gauche / droite
# ---------------------------------------------------------------------------


@dataclass
class RedondanceGD:
    moyenne_pc_pe: float = float("nan")
    ecart_type_pc_pe: float = float("nan")
    max_absolu_pc_pe: float = float("nan")
    moyenne_Nm: float = float("nan")
    ecart_type_Nm: float = float("nan")
    n: int = 0
    non_calculable: str | None = None


def redondance_gauche_droite(
    gauche: np.ndarray | None,
    droite: np.ndarray | None,
    pleine_echelle_Nm: float,
) -> RedondanceGD:
    """Résidu gauche − droite : indicateur indépendant de la référence banc.

    HYPOTHÈSE : n'a de sens que sur un essai en ligne droite sans sollicitation
    différentielle (déclaré par `ligne_droite: true` en configuration), où les
    deux transmissions voient physiquement le même couple. En virage ou sous
    action d'un différentiel piloté, l'écart mesuré est réel et non métrologique.
    """
    if gauche is None or droite is None:
        return RedondanceGD(
            non_calculable="résidu gauche−droite non calculable : les deux voies de couple "
            "ne sont pas disponibles simultanément sur cet essai"
        )
    d = np.asarray(gauche, dtype=float) - np.asarray(droite, dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return RedondanceGD(non_calculable="résidu gauche−droite non calculable : pas assez d'échantillons valides")
    return RedondanceGD(
        moyenne_pc_pe=float(100.0 * d.mean() / pleine_echelle_Nm),
        ecart_type_pc_pe=float(100.0 * d.std(ddof=1) / pleine_echelle_Nm),
        max_absolu_pc_pe=float(100.0 * np.abs(d).max() / pleine_echelle_Nm),
        moyenne_Nm=float(d.mean()),
        ecart_type_Nm=float(d.std(ddof=1)),
        n=int(d.size),
    )


# ---------------------------------------------------------------------------
# Bilan d'incertitude
# ---------------------------------------------------------------------------

# Diviseurs de conversion d'une demi-étendue en incertitude-type (GUM).
DIVISEURS = {"normale": 1.0, "rectangulaire": math.sqrt(3.0), "triangulaire": math.sqrt(6.0)}


@dataclass
class Contribution:
    nom: str
    valeur_Nm: float
    loi: str
    commentaire: str = ""

    @property
    def u_Nm(self) -> float:
        return abs(self.valeur_Nm) / DIVISEURS[self.loi]


@dataclass
class BilanIncertitude:
    contributions: list[Contribution] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)
    u_composee_Nm: float = float("nan")
    U_k2_Nm: float = float("nan")
    U_k2_pc_pe: float = float("nan")
    minorant: bool = False
    non_calculable: str | None = None


def bilan_incertitude(
    contributions: Sequence[Contribution],
    exclusions: Sequence[str],
    pleine_echelle_Nm: float,
) -> BilanIncertitude:
    """Combinaison quadratique des contributions et incertitude élargie k=2.

    HYPOTHÈSES :
      * les contributions sont supposées **non corrélées** entre elles, ce qui
        autorise la somme quadratique (GUM, § 5.1.2) ;
      * une grandeur bornée dont on ne connaît que la demi-étendue est traitée
        en loi rectangulaire (u = a/√3) : non-linéarité, hystérésis, dérive de
        zéro, effet thermique ;
      * la répétabilité est une évaluation de type A, donc directement un
        écart-type (u = s) ;
      * k = 2, soit un niveau de confiance d'environ 95 % sous hypothèse de
        distribution résultante approximativement normale.

    Toute contribution non disponible est **exclue et signalée** : le résultat
    est alors explicitement annoncé comme un **minorant** de l'incertitude.
    """
    retenues = [c for c in contributions if math.isfinite(c.valeur_Nm)]
    if not retenues:
        return BilanIncertitude(
            exclusions=list(exclusions),
            minorant=True,
            non_calculable="bilan d'incertitude non calculable : aucune contribution disponible",
        )
    u_c = math.sqrt(sum(c.u_Nm**2 for c in retenues))
    U = 2.0 * u_c
    return BilanIncertitude(
        contributions=retenues,
        exclusions=list(exclusions),
        u_composee_Nm=u_c,
        U_k2_Nm=U,
        U_k2_pc_pe=100.0 * U / pleine_echelle_Nm,
        minorant=bool(exclusions),
    )


# ---------------------------------------------------------------------------
# Cartes de contrôle (chapitre 11)
# ---------------------------------------------------------------------------


@dataclass
class ParametresSPC:
    mu0_pc_pe: float = float("nan")
    sigma0_pc_pe: float = float("nan")
    sigma0_robuste_pc_pe: float = float("nan")
    n: int = 0
    ddl: int = 0
    # CUSUM
    cusum_k_pc_pe: float = float("nan")
    cusum_h_alerte_pc_pe: float = float("nan")
    cusum_h_alarme_pc_pe: float = float("nan")
    # EWMA
    ewma_lambda: float = float("nan")
    ewma_L_alerte: float = float("nan")
    ewma_L_alarme: float = float("nan")
    ewma_sigma_asymptotique_pc_pe: float = float("nan")
    ewma_limite_alerte_pc_pe: float = float("nan")
    ewma_limite_alarme_pc_pe: float = float("nan")
    non_calculable: str | None = None


# Constante d2 de l'étendue mobile de portée 2 (tables de maîtrise statistique
# des procédés) : sigma ≈ MR_moyen / d2. Sert d'estimateur robuste, insensible
# à une dérive lente qui gonflerait l'écart-type classique.
D2_ETENDUE_MOBILE = 1.128


def parametres_spc(
    echantillons_pc_pe: Sequence[float],
    ddl: int = 0,
    lam: float = 0.2,
    sigma0_pc_pe: float | None = None,
) -> ParametresSPC:
    """Déduit μ0, σ0 et les seuils CUSUM / EWMA de la dispersion mesurée.

    μ0 et σ0 proviennent **exclusivement des essais répétés** ; aucune valeur
    n'est posée a priori.

    HYPOTHÈSE IMPORTANTE sur σ0 : la dispersion à porter dans la carte est la
    répétabilité **à couple de référence constant** (`sigma0_pc_pe`, écart-type
    poolé intra-niveau), et non l'écart-type brut des résidus toutes conditions
    confondues. Dès qu'il existe une erreur de gain, le résidu varie avec le
    couple appliqué : l'écart-type brut mesurerait alors l'étendue de la plage
    d'essai bien plus que la variabilité de la chaîne de mesure, et les limites
    de contrôle en seraient très largement surdimensionnées. À défaut de valeur
    fournie, l'écart-type brut est utilisé et le rapport le signale.

    Les couples (λ, L) et les multiplicateurs CUSUM ne sont pas arbitraires non
    plus : ce sont les valeurs tabulées qui donnent les ARL0 (longueur moyenne
    de série sous contrôle) usuelles en maîtrise statistique des procédés
    (Montgomery, *Introduction to Statistical Quality Control*) :

      * CUSUM  k = 0,5·σ0 → carte optimisée pour détecter un décalage de 1 σ ;
                h = 4·σ0  → ARL0 ≈ 168 (seuil d'**alerte**) ;
                h = 5·σ0  → ARL0 ≈ 465 (seuil d'**alarme**).
      * EWMA   λ = 0,20 avec L = 2,962 → ARL0 ≈ 370 (seuil d'**alarme**) ;
               L = 2,0 → ≈ 2 σ, seuil d'**alerte** conventionnel.
        σ_EWMA(∞) = σ0·√(λ / (2 − λ)).
    """
    x = np.asarray([v for v in echantillons_pc_pe if math.isfinite(v)], dtype=float)
    if x.size < 2:
        return ParametresSPC(
            n=int(x.size),
            non_calculable=(
                "paramètres SPC non calculables : la dispersion doit être estimée sur des essais "
                f"répétés, or {x.size} valeur(s) exploitable(s) seulement"
            ),
        )

    mu0 = float(x.mean())
    sigma0 = float(sigma0_pc_pe) if sigma0_pc_pe is not None else float(x.std(ddof=1))
    etendues = np.abs(np.diff(x))
    sigma_robuste = float(etendues.mean() / D2_ETENDUE_MOBILE) if etendues.size else float("nan")

    if sigma0 == 0:
        return ParametresSPC(
            mu0_pc_pe=mu0,
            sigma0_pc_pe=0.0,
            n=int(x.size),
            non_calculable="paramètres SPC non calculables : dispersion nulle sur les essais répétés",
        )

    L_alarme, L_alerte = 2.962, 2.0
    sigma_ewma = sigma0 * math.sqrt(lam / (2.0 - lam))
    return ParametresSPC(
        mu0_pc_pe=mu0,
        sigma0_pc_pe=sigma0,
        sigma0_robuste_pc_pe=sigma_robuste,
        n=int(x.size),
        ddl=ddl or int(x.size - 1),
        cusum_k_pc_pe=0.5 * sigma0,
        cusum_h_alerte_pc_pe=4.0 * sigma0,
        cusum_h_alarme_pc_pe=5.0 * sigma0,
        ewma_lambda=lam,
        ewma_L_alerte=L_alerte,
        ewma_L_alarme=L_alarme,
        ewma_sigma_asymptotique_pc_pe=sigma_ewma,
        ewma_limite_alerte_pc_pe=L_alerte * sigma_ewma,
        ewma_limite_alarme_pc_pe=L_alarme * sigma_ewma,
    )


# ---------------------------------------------------------------------------
# Diagnostic qualitatif
# ---------------------------------------------------------------------------


def correlation_simple(a: np.ndarray, b: np.ndarray, decimation: int = 100) -> float:
    """Coefficient de corrélation de Pearson entre deux signaux, après décimation.

    La décimation réduit l'autocorrélation des séries à haute cadence. Seule la
    **valeur** de r est exploitée (jamais une p-value, qui n'aurait pas de sens
    sur des échantillons non indépendants).
    """
    a = np.asarray(a, dtype=float)[::decimation]
    b = np.asarray(b, dtype=float)[::decimation]
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@dataclass
class Diagnostic:
    cause: str
    conclusion: str
    indices: dict[str, float] = field(default_factory=dict)


def diagnostiquer(
    nom_essai: str,
    reg: Regression | None,
    recal: Recalage | None,
    correlation_residu_regime: float,
    correlation_residu_couple: float,
    pleine_echelle_Nm: float,
    params: ParamsDiagnostic,
    reproductible: bool | None = None,
) -> Diagnostic:
    """Applique la grille d'interprétation demandée, dans cet ordre de priorité.

    Grille :
      * écart uniquement en transitoire            → **synchronisation** ;
      * écart proportionnel au couple              → **gain** ;
      * écart constant                             → **offset capteur** ;
      * écart dépendant du point de fonctionnement
        et reproductible d'un essai à l'autre      → **écart de modèle**.

    La synchronisation est testée en premier : un défaut de synchronisation
    produit mécaniquement un résidu qui *ressemble* à un défaut de gain sur un
    signal transitoire, alors que l'inverse n'est pas vrai.
    """
    indices: dict[str, float] = {}
    offset_pc_pe = 100.0 * reg.b / pleine_echelle_Nm if reg and not reg.non_calculable else float("nan")
    gain_pc = 100.0 * (reg.a - 1.0) if reg and not reg.non_calculable else float("nan")
    if math.isfinite(offset_pc_pe):
        indices["offset_pc_pe"] = offset_pc_pe
    if math.isfinite(gain_pc):
        indices["gain_pc"] = gain_pc
    if recal and not recal.non_calculable:
        indices["retard_ms"] = recal.retard_ms
        indices["gain_rms_recalage"] = recal.gain_rms
    indices["r_residu_regime"] = correlation_residu_regime
    indices["r_residu_couple"] = correlation_residu_couple

    # Un retard significatif peut coexister avec un offset ou un défaut de gain
    # sans être la cause dominante du résidu. Il est alors signalé en incise
    # dans la conclusion, pour ne pas le passer sous silence tout en tenant la
    # limite de trois phrases.
    retard_notable = (
        recal is not None
        and not recal.non_calculable
        and abs(recal.retard_ms) > params.seuil_retard_ms
    )

    # 1. Synchronisation
    if (
        recal
        and not recal.non_calculable
        and abs(recal.retard_ms) > params.seuil_retard_ms
        and math.isfinite(recal.gain_rms)
        and recal.gain_rms > params.gain_rms_recalage
    ):
        # Un écart de gain reste possible en parallèle : il est identifié sur le
        # balayage statique, insensible par construction au défaut de
        # synchronisation. On le mentionne sans changer la cause dominante.
        reste = (
            f", l'écart de sensibilité de {gain_pc:+.2f} % relevé au balayage statique restant "
            f"à traiter séparément"
            if math.isfinite(gain_pc) and abs(gain_pc) > params.seuil_gain_pc
            else ""
        )
        return Diagnostic(
            "synchronisation",
            f"Sur {nom_essai}, l'écart entre le couple mesuré par les transmissions et la "
            f"référence banc se concentre dans les phases transitoires : un recalage temporel de "
            f"{recal.retard_ms:.0f} ms réduit le RMS du résidu de "
            f"{100 * recal.gain_rms:.0f} %. L'écart relève donc de la synchronisation des voies "
            f"d'acquisition et non d'un défaut de la chaîne de mesure de couple. "
            f"Après recalage, le résidu résiduel retombe à {recal.rms_residu_apres_Nm:.1f} N·m "
            f"RMS{reste}.",
            indices,
        )

    # 2. Gain
    if math.isfinite(gain_pc) and abs(gain_pc) > params.seuil_gain_pc:
        return Diagnostic(
            "gain",
            f"Sur {nom_essai}, l'écart croît proportionnellement au couple appliqué : la pente de "
            f"régression vaut {reg.a:.4f}, soit une erreur de sensibilité de {gain_pc:+.2f} %. "
            f"Le comportement signe une erreur de gain de la chaîne de mesure (étalonnage de la "
            f"jauge ou du conditionnement télémétrique) et non un décalage de zéro. "
            f"Une correction de gain sur la voie transmissions est à envisager"
            + (
                f", ainsi qu'un recalage de {recal.retard_ms:+.0f} ms des voies d'acquisition, "
                f"qui ne réduit toutefois le résidu que de {100 * recal.gain_rms:.0f} %."
                if retard_notable
                else "."
            ),
            indices,
        )

    # 3. Offset
    if math.isfinite(offset_pc_pe) and abs(offset_pc_pe) > params.seuil_offset_pc_pe:
        return Diagnostic(
            "offset",
            f"Sur {nom_essai}, l'écart au couple de référence est sensiblement constant sur toute "
            f"la plage, à {reg.b:+.1f} N·m ({offset_pc_pe:+.2f} % PE), pour une pente de "
            f"{reg.a:.4f} proche de l'unité. Il s'agit d'un offset de la chaîne de mesure, "
            f"corrigeable par une remise à zéro avant essai. "
            f"L'aptitude en sensibilité n'est pas remise en cause"
            + (
                f", un retard de {recal.retard_ms:+.0f} ms subsistant néanmoins entre les voies "
                f"d'acquisition."
                if retard_notable
                else "."
            ),
            indices,
        )

    # 4. Écart de modèle
    r_max = max(
        (abs(v) for v in (correlation_residu_regime, correlation_residu_couple) if math.isfinite(v)),
        default=float("nan"),
    )
    if math.isfinite(r_max) and r_max > params.seuil_correlation_point_fct:
        mention = (
            " et se reproduit d'un essai à l'autre"
            if reproductible
            else (" mais sa reproductibilité d'un essai à l'autre n'a pas pu être vérifiée"
                  if reproductible is None else " sans se reproduire d'un essai à l'autre")
        )
        return Diagnostic(
            "modele" if reproductible else "point_de_fonctionnement",
            f"Sur {nom_essai}, le résidu ne présente ni offset ni erreur de gain significatifs mais "
            f"dépend du point de fonctionnement (|r| = {r_max:.2f} avec le régime ou le couple)"
            f"{mention}. Ce comportement oriente vers un écart de modèle entre la grandeur "
            f"reconstruite au banc et le couple réellement transmis, plutôt que vers un défaut de "
            f"la chaîne de mesure. "
            f"Aucune correction de la voie transmissions n'est justifiée à ce stade.",
            indices,
        )

    return Diagnostic(
        "conforme",
        f"Sur {nom_essai}, le couple mesuré par les transmissions suit la référence banc sans "
        f"offset, erreur de gain ni retard significatifs au regard des seuils retenus "
        f"({params.seuil_offset_pc_pe} % PE, {params.seuil_gain_pc} %, "
        f"{params.seuil_retard_ms} ms). "
        f"La corrélation est jugée satisfaisante sur cet essai.",
        indices,
    )
