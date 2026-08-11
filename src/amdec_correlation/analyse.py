"""Orchestration : parcourt les essais, applique le traitement de chaque type.

Chaque essai est traité selon son `type` déclaré en configuration :

  * `balayage`     → régression, offset, sensibilité, non-linéarité, hystérésis ;
  * `repetabilite` → dispersion à couple de référence constant ;
  * `dynamique`    → recalage temporel par intercorrélation.

Les traitements transverses (sensibilité thermique, dérive du zéro, redondance
gauche/droite) sont appliqués à tous les essais qui disposent des canaux
nécessaires, quel que soit leur type.

Aucun résultat n'est produit par défaut : chaque grandeur non calculable porte
le motif exact de sa non-calculabilité, restitué tel quel dans le rapport.
"""

from __future__ import annotations

import math
import re
import unicodedata
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import graphiques, metriques as M
from .config import Config, DeclarationEssai
from .io_mdf import (
    ErreurChargement,
    SignauxEssai,
    charger_signaux,
    lister_fichiers,
    resumer_etats,
)

# Cadence de conservation des échantillons pour l'analyse thermique groupée :
# la sensibilité thermique est une tendance lente, 10 Hz suffisent largement et
# cela borne l'empreinte mémoire quand on met en commun toute la campagne.
FREQUENCE_THERMIQUE_Hz = 10.0


def slug(texte: str) -> str:
    """Nom de fichier sûr à partir d'un libellé d'essai accentué."""
    sans_accent = "".join(
        c for c in unicodedata.normalize("NFD", texte) if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"[^a-zA-Z0-9]+", "_", sans_accent).strip("_").lower()


# ---------------------------------------------------------------------------
# Données d'un fichier, prêtes à l'exploitation
# ---------------------------------------------------------------------------


@dataclass
class DonneesFichier:
    chemin: Path
    t: np.ndarray
    reference: np.ndarray  # déjà ramenée au point de mesure des transmissions
    mesure: np.ndarray  # voie de comparaison choisie selon `comparaison.mode`
    gauche: np.ndarray | None
    droite: np.ndarray | None
    regime: np.ndarray | None
    temperature: np.ndarray | None
    signaux: SignauxEssai = field(repr=False)

    @property
    def residu(self) -> np.ndarray:
        return self.mesure - self.reference


def _combiner(
    mode: str, gauche, droite, unique, grandeur: str
) -> tuple[np.ndarray, str]:
    """Construit le signal comparé à partir des voies disponibles.

    Le mode s'applique **de la même façon au couple mesuré et au couple de
    référence** : comparer la moyenne de deux voies mesurées à la somme de deux
    voies de référence n'aurait aucun sens.

    Ordre de résolution : si les deux voies gauche/droite existent, le mode
    tranche ; sinon on retombe sur la voie unique, ou sur la seule voie latérale
    disponible.
    """
    if gauche is not None and droite is not None:
        if mode == "gauche":
            return gauche, f"{grandeur} : voie gauche"
        if mode == "droite":
            return droite, f"{grandeur} : voie droite"
        if mode == "somme":
            return gauche + droite, f"{grandeur} : somme des deux voies"
        return 0.5 * (gauche + droite), f"{grandeur} : moyenne des deux voies"

    if mode in ("gauche", "droite"):
        choisie = gauche if mode == "gauche" else droite
        if choisie is not None:
            return choisie, f"{grandeur} : voie {mode}"

    repli = unique if unique is not None else (gauche if gauche is not None else droite)
    if repli is None:
        raise ErreurChargement(f"aucune voie disponible pour le {grandeur}.")
    if gauche is not None or droite is not None:
        cote = "gauche" if gauche is not None else "droite"
        warnings.warn(
            f"mode « {mode} » demandé pour le {grandeur} mais une seule voie ({cote}) "
            "est disponible : cette voie est utilisée telle quelle.",
            stacklevel=2,
        )
        return repli, f"{grandeur} : voie {cote} (seule voie disponible)"
    return repli, f"{grandeur} : voie unique"


def preparer(chemin: Path, cfg: Config, mapping) -> DonneesFichier:
    """Charge un fichier et construit les signaux de travail.

    HYPOTHÈSE : le couple de référence est multiplié par
    `comparaison.rapport_reduction` pour être ramené au point de mesure des
    transmissions. Avec un rapport de 1,0 (référence prise au même point,
    configuration retenue pour cette campagne) l'opération est neutre.
    """
    signaux = charger_signaux(chemin, mapping, cfg.frequence_Hz)
    gauche = signaux.a("couple_mesure_gauche")
    droite = signaux.a("couple_mesure_droite")
    mesure, _ = _combiner(cfg.mode_comparaison, gauche, droite, None, "couple mesuré")
    reference, _ = _combiner(
        cfg.mode_comparaison,
        signaux.a("couple_reference_gauche"),
        signaux.a("couple_reference_droite"),
        signaux.a("couple_reference"),
        "couple de référence",
    )
    return DonneesFichier(
        chemin=chemin,
        t=signaux.t,
        reference=reference * cfg.rapport_reduction,
        mesure=mesure,
        gauche=gauche,
        droite=droite,
        regime=signaux.a("regime"),
        temperature=signaux.a("temperature"),
        signaux=signaux,
    )


# ---------------------------------------------------------------------------
# Résultats
# ---------------------------------------------------------------------------


@dataclass
class ResultatEssai:
    declaration: DeclarationEssai
    fichiers_traites: list[Path] = field(default_factory=list)
    erreurs: list[str] = field(default_factory=list)
    paliers: list[M.Palier] = field(default_factory=list)
    regression: M.Regression | None = None
    hysteresis: M.Hysteresis | None = None
    non_linearite_pc_pe: float = float("nan")
    repetabilite: M.Repetabilite | None = None
    recalages: list[tuple[Path, M.Recalage]] = field(default_factory=list)
    thermique: M.SensibiliteThermique | None = None
    derives_zero: list[tuple[Path, M.DeriveZero]] = field(default_factory=list)
    redondance: M.RedondanceGD | None = None
    diagnostic: M.Diagnostic | None = None
    r_residu_regime: float = float("nan")
    r_residu_couple: float = float("nan")
    diagnostics_temps: list[str] = field(default_factory=list)
    etats: dict[str, dict[str, float]] = field(default_factory=dict)
    figures: list[Path] = field(default_factory=list)
    # Échantillons décimés conservés pour l'analyse thermique, y compris au
    # niveau de la campagne. Ce sont les données BRUTES (non corrigées de
    # l'effet du couple) : la correction est faite une seule fois, au moment de
    # l'ajustement, pour ne pas être appliquée deux fois lors de la mise en
    # commun de tous les essais.
    pool_temperature: np.ndarray | None = field(default=None, repr=False)
    pool_residu: np.ndarray | None = field(default=None, repr=False)
    pool_couple: np.ndarray | None = field(default=None, repr=False)

    @property
    def nom(self) -> str:
        return self.declaration.dossier

    @property
    def recalage_principal(self) -> M.Recalage | None:
        """Recalage retenu : celui dont le pic d'intercorrélation est le plus marqué."""
        valides = [(c, r) for c, r in self.recalages if not r.non_calculable]
        if not valides:
            return self.recalages[0][1] if self.recalages else None
        return max(valides, key=lambda cr: cr[1].correlation_pic)[1]

    @property
    def derives_zero_valides(self) -> list[M.DeriveZero]:
        return [d for _, d in self.derives_zero if not d.non_calculable]


@dataclass
class ResultatCampagne:
    config: Config
    essais: list[ResultatEssai] = field(default_factory=list)
    thermique_globale: M.SensibiliteThermique | None = None
    spc: M.ParametresSPC | None = None
    repetabilite_remontage: M.Repetabilite | None = None
    motif_remontage: str | None = None
    incertitude: M.BilanIncertitude | None = None
    avertissements: list[str] = field(default_factory=list)
    figures: list[Path] = field(default_factory=list)

    def par_type(self, type_essai: str) -> list[ResultatEssai]:
        return [e for e in self.essais if e.declaration.type == type_essai]

    def essai(self, dossier: str) -> ResultatEssai | None:
        return next((e for e in self.essais if e.nom == dossier), None)


# ---------------------------------------------------------------------------
# Traitement d'un essai
# ---------------------------------------------------------------------------


def traiter_essai(essai: DeclarationEssai, cfg: Config, dossier_figures: Path) -> ResultatEssai:
    resultat = ResultatEssai(declaration=essai)
    mapping = cfg.canaux_pour(essai.dossier)
    chemin = cfg.chemin_essai(essai.dossier)
    fichiers = lister_fichiers(chemin, essai.max_fichiers)

    if not fichiers:
        resultat.erreurs.append(f"aucun fichier .mf4 trouvé dans {chemin}")
        return resultat
    if not mapping.a_reference or not mapping.voies_couple_mesure:
        manquant = "couple de référence" if not mapping.a_reference else "couple mesuré"
        resultat.erreurs.append(
            f"mapping incomplet ({manquant} non renseigné) : essai non exploitable"
        )
        return resultat

    # Accumulateurs inter-fichiers.
    T_pool: list[np.ndarray] = []
    res_pool: list[np.ndarray] = []
    couple_pool: list[np.ndarray] = []
    r_regime: list[float] = []
    r_couple: list[float] = []
    donnees_figure: DonneesFichier | None = None
    meilleur_pic = -np.inf

    for fichier in fichiers:
        try:
            d = preparer(fichier, cfg, mapping)
        except Exception as exc:  # un fichier illisible n'interrompt pas la campagne
            resultat.erreurs.append(f"{fichier.name} : {exc}")
            continue

        resultat.fichiers_traites.append(fichier)
        resultat.diagnostics_temps.append(f"{fichier.name} — {d.signaux.diagnostic.resume()}")
        if d.signaux.etats:
            resultat.etats[fichier.name] = resumer_etats(d.signaux)

        # -- paliers stabilisés (balayage et répétabilité) ------------------
        if essai.type in ("balayage", "repetabilite"):
            resultat.paliers.extend(
                M.detecter_paliers(
                    d.t, d.reference, d.mesure, cfg.pleine_echelle_Nm, cfg.paliers,
                    temperature=d.temperature, source=fichier.name,
                )
            )

        # -- recalage temporel (essais dynamiques) -------------------------
        # `residu_thermique` sert à la régression sur la température. Sur un
        # essai dynamique, le résidu brut contient l'erreur induite par le
        # décalage temporel, qui culmine dans les transitoires de couple et
        # dépasse largement l'effet thermique recherché. On régresse donc sur
        # le résidu APRÈS recalage, dont ces transitoires sont retirés.
        residu_thermique = d.residu
        if essai.type == "dynamique":
            recal = M.recalage_temporel(
                d.t, d.reference, d.mesure, cfg.pleine_echelle_Nm, cfg.intercorrelation
            )
            resultat.recalages.append((fichier, recal))
            if not recal.non_calculable:
                residu_thermique = (
                    M.appliquer_retard(d.t, d.mesure, recal.retard_ms / 1000.0) - d.reference
                )
            pic = recal.correlation_pic if not recal.non_calculable else -np.inf
            if pic > meilleur_pic:
                meilleur_pic, donnees_figure = pic, d

        # -- dérive du zéro ------------------------------------------------
        resultat.derives_zero.append(
            (
                fichier,
                M.derive_zero(
                    d.t, d.mesure, d.reference, d.regime, cfg.pleine_echelle_Nm, cfg.zero
                ),
            )
        )

        # -- accumulation thermique (décimée) ------------------------------
        if d.temperature is not None:
            pas = max(1, int(round(cfg.frequence_Hz / FREQUENCE_THERMIQUE_Hz)))
            T_pool.append(d.temperature[::pas])
            res_pool.append(residu_thermique[::pas])
            couple_pool.append(d.reference[::pas])

        # -- corrélations résidu / point de fonctionnement ------------------
        if d.regime is not None:
            r_regime.append(M.correlation_simple(d.residu, d.regime))
        r_couple.append(M.correlation_simple(d.residu, d.reference))

        # -- redondance gauche / droite ------------------------------------
        if essai.ligne_droite and resultat.redondance is None:
            resultat.redondance = M.redondance_gauche_droite(
                d.gauche, d.droite, cfg.pleine_echelle_Nm
            )
            if not resultat.redondance.non_calculable:
                resultat.figures.append(
                    graphiques.figure_redondance(
                        d.t, d.gauche, d.droite, cfg.pleine_echelle_Nm,
                        dossier_figures / f"{slug(essai.dossier)}_redondance_gd.png",
                        titre=f"{essai.dossier} — redondance des voies (gauche − droite)",
                    )
                )
        if donnees_figure is None:
            donnees_figure = d

    if not resultat.fichiers_traites:
        return resultat

    # -- exploitation des paliers ------------------------------------------
    if resultat.paliers:
        reg = M.regression(
            [p.reference for p in resultat.paliers], [p.mesure for p in resultat.paliers]
        )
        resultat.regression = reg
        resultat.non_linearite_pc_pe = M.non_linearite_pc_pe(reg, cfg.pleine_echelle_Nm)
        resultat.hysteresis = M.hysteresis(resultat.paliers, cfg.pleine_echelle_Nm, cfg.paliers)
        resultat.repetabilite = M.repetabilite(
            resultat.paliers, cfg.pleine_echelle_Nm, cfg.paliers
        )

        if essai.type == "balayage":
            resultat.figures.append(
                graphiques.figure_regression_balayage(
                    resultat.paliers, reg, resultat.hysteresis, cfg.pleine_echelle_Nm,
                    dossier_figures / f"{slug(essai.dossier)}_regression.png",
                    titre=f"{essai.dossier} — corrélation transmissions / banc GMP",
                )
            )
        if essai.type == "repetabilite":
            resultat.figures.append(
                graphiques.figure_repetabilite(
                    resultat.repetabilite, cfg.pleine_echelle_Nm,
                    dossier_figures / f"{slug(essai.dossier)}_repetabilite.png",
                    titre=f"{essai.dossier} — répétabilité du résidu",
                )
            )

    # -- sensibilité thermique de l'essai -----------------------------------
    if T_pool:
        resultat.pool_temperature = np.concatenate(T_pool)
        resultat.pool_residu = np.concatenate(res_pool)
        resultat.pool_couple = np.concatenate(couple_pool)
        resultat.thermique = M.sensibilite_thermique(
            resultat.pool_temperature, resultat.pool_residu,
            cfg.pleine_echelle_Nm, cfg.thermique, couple=resultat.pool_couple,
        )

    # -- recalage : figure sur le fichier le plus contrasté ------------------
    if essai.type == "dynamique" and donnees_figure is not None:
        recal = resultat.recalage_principal
        if recal is not None:
            resultat.figures.append(
                graphiques.figure_recalage(
                    donnees_figure.t, donnees_figure.reference, donnees_figure.mesure,
                    recal, dossier_figures / f"{slug(essai.dossier)}_recalage.png",
                    titre=essai.dossier,
                )
            )

    resultat.r_residu_regime = float(np.nanmean(r_regime)) if r_regime else float("nan")
    resultat.r_residu_couple = float(np.nanmean(r_couple)) if r_couple else float("nan")
    return resultat


# ---------------------------------------------------------------------------
# Traitements transverses à la campagne
# ---------------------------------------------------------------------------


def _repetabilite_remontage(
    campagne: ResultatCampagne, cfg: Config
) -> tuple[M.Repetabilite | None, str | None]:
    """Répétabilité entre groupes de remontage.

    Ne peut être calculée que si au moins deux essais portent des étiquettes
    `groupe_remontage` différentes : sans repère avant/après démontage, la
    grandeur n'est pas définie et le rapport le dira explicitement.
    """
    groupes: dict[str, list[M.Palier]] = {}
    for e in campagne.essais:
        etiquette = e.declaration.groupe_remontage
        if etiquette and e.paliers:
            groupes.setdefault(etiquette, []).extend(e.paliers)

    if len(groupes) < 2:
        # Deux situations très différentes mènent à « non calculable », et le
        # rapport ne doit pas les confondre : une grandeur qui n'existe pas pour
        # cette campagne n'est pas une grandeur qu'on a oublié de configurer.
        if cfg.remontage_realise is False:
            return None, (
                "répétabilité après remontage non calculable : aucun démontage ni remontage "
                "de la chaîne de mesure n'a été réalisé au cours de la campagne. La grandeur "
                "n'est pas définie ici — elle n'est pas manquante, et aucune valeur ne peut "
                "lui être substituée"
            )
        return None, (
            "répétabilité après remontage non calculable : "
            f"{len(groupes)} groupe(s) de remontage déclaré(s) en configuration "
            "(clé `groupe_remontage` sur les essais), il en faut au moins 2 "
            "(ex. « avant » et « apres ») pour comparer un état à l'autre"
        )

    # Un « palier moyen par niveau et par groupe » : la dispersion entre groupes
    # à même niveau de couple est l'écart lié au remontage.
    tolerance = cfg.paliers.tolerance_appariement_pc_pe * cfg.pleine_echelle_Nm / 100.0
    representatifs: list[M.Palier] = []
    for etiquette, paliers in groupes.items():
        tries = sorted(paliers, key=lambda p: p.reference)
        paquets: list[list[M.Palier]] = [[tries[0]]]
        for p in tries[1:]:
            if abs(p.reference - paquets[-1][0].reference) <= tolerance:
                paquets[-1].append(p)
            else:
                paquets.append([p])
        for paquet in paquets:
            representatifs.append(
                M.Palier(
                    t_debut=paquet[0].t_debut,
                    t_fin=paquet[-1].t_fin,
                    reference=float(np.mean([p.reference for p in paquet])),
                    mesure=float(np.mean([p.mesure for p in paquet])),
                    ecart_type_mesure=float(np.std([p.mesure for p in paquet], ddof=0)),
                    ecart_type_reference=0.0,
                    n=len(paquet),
                    source=etiquette,
                )
            )
    return M.repetabilite(representatifs, cfg.pleine_echelle_Nm, cfg.paliers), None


def _residus_centres_par_niveau(campagne: ResultatCampagne, cfg: Config) -> list[float]:
    """Résidus des essais répétés (% PE), centrés par essai ET par niveau de couple.

    Le sous-groupe de centrage est le couple (essai, niveau de couple). La
    dispersion de la série obtenue est donc la variabilité **court terme** de la
    chaîne de mesure : à point de fonctionnement donné et sans démontage. C'est
    la définition classique du σ0 d'une carte de contrôle — les décalages entre
    essais ou entre remontages sont précisément ce que la carte doit détecter,
    et non ce qui doit élargir ses limites.

    La moyenne d'ensemble est réinjectée pour que μ0 reste le résidu moyen
    effectivement observé sur la campagne.
    """
    essais = campagne.par_type("repetabilite")
    tous = [p for e in essais for p in e.paliers]
    if not tous:
        return []
    tolerance = cfg.paliers.tolerance_appariement_pc_pe * cfg.pleine_echelle_Nm / 100.0
    moyenne_globale = float(np.mean([p.residu for p in tous]))

    centres: list[float] = []
    for essai in essais:
        if not essai.paliers:
            continue
        tries = sorted(essai.paliers, key=lambda p: p.reference)
        groupes: list[list[M.Palier]] = [[tries[0]]]
        for p in tries[1:]:
            if abs(p.reference - groupes[-1][0].reference) <= tolerance:
                groupes[-1].append(p)
            else:
                groupes.append([p])
        for groupe in groupes:
            moyenne_groupe = float(np.mean([p.residu for p in groupe]))
            centres.extend(
                100.0 * (p.residu - moyenne_groupe + moyenne_globale) / cfg.pleine_echelle_Nm
                for p in groupe
            )
    return centres


def _incertitude(campagne: ResultatCampagne, cfg: Config) -> M.BilanIncertitude:
    """Assemble le bilan d'incertitude à partir des grandeurs effectivement mesurées."""
    contributions: list[M.Contribution] = []
    exclusions: list[str] = []
    pe = cfg.pleine_echelle_Nm

    balayages = campagne.par_type("balayage")
    reg_essai = next((e for e in balayages if e.regression and not e.regression.non_calculable), None)

    if reg_essai and np.isfinite(reg_essai.non_linearite_pc_pe):
        contributions.append(
            M.Contribution(
                "Non-linéarité", cfg.nm(reg_essai.non_linearite_pc_pe), "rectangulaire",
                f"résidu max de la régression, essai « {reg_essai.nom} »",
            )
        )
    else:
        exclusions.append("non-linéarité (aucun essai de balayage exploitable)")

    hyst = next(
        (e.hysteresis for e in balayages if e.hysteresis and not e.hysteresis.non_calculable), None
    )
    if hyst:
        # Demi-étendue : l'hystérésis est un écart crête à crête entre montée et
        # descente, la valeur possible autour de la moyenne vaut donc H/2.
        contributions.append(
            M.Contribution("Hystérésis", cfg.nm(hyst.max_pc_pe) / 2.0, "rectangulaire",
                           "demi-écart montée/descente")
        )
    else:
        exclusions.append("hystérésis (pas d'appariement montée/descente disponible)")

    rep = next(
        (e.repetabilite for e in campagne.par_type("repetabilite")
         if e.repetabilite and not e.repetabilite.non_calculable),
        None,
    )
    if rep:
        contributions.append(
            M.Contribution("Répétabilité", rep.ecart_type_Nm, "normale",
                           f"écart-type poolé, {rep.degres_liberte} ddl")
        )
    else:
        exclusions.append("répétabilité (aucun essai répété exploitable)")

    derives = [d.derive_pc_pe for e in campagne.essais for d in e.derives_zero_valides]
    if derives:
        pire = max(derives, key=abs)
        contributions.append(
            M.Contribution("Dérive de zéro", cfg.nm(pire), "rectangulaire",
                           f"dérive la plus forte sur {len(derives)} essai(s)")
        )
    else:
        exclusions.append("dérive de zéro (aucun relevé de zéro avant/après identifiable)")

    th = campagne.thermique_globale
    if th and not th.non_calculable:
        if cfg.thermique.plage_service_C:
            # Demi-étendue de l'effet thermique sur la plage de service.
            demi = abs(th.pente_Nm_par_C) * cfg.thermique.plage_service_C / 2.0
            contributions.append(
                M.Contribution("Effet thermique", demi, "rectangulaire",
                               f"sur une plage de service de {cfg.thermique.plage_service_C:.0f} °C")
            )
        else:
            exclusions.append(
                "effet thermique (sensibilité mesurée, mais `thermique.plage_service_C` "
                "n'est pas renseignée en configuration)"
            )
    else:
        exclusions.append("effet thermique (sensibilité thermique non calculable)")

    if cfg.incertitude_reference_k1_Nm is not None:
        contributions.append(
            M.Contribution("Référence banc GMP", cfg.incertitude_reference_k1_Nm, "normale",
                           "incertitude-type du moyen de référence, fournie en configuration")
        )
    else:
        exclusions.append(
            "incertitude du moyen de référence (`comparaison.incertitude_reference_k1_Nm` "
            "non renseignée : elle ne peut pas être déduite des acquisitions)"
        )

    return M.bilan_incertitude(contributions, exclusions, pe)


def _diagnostics(campagne: ResultatCampagne, cfg: Config) -> None:
    """Applique la grille qualitative, après avoir jugé de la reproductibilité."""
    dynamiques = campagne.par_type("dynamique")
    # Reproductible : au moins deux essais dynamiques présentent une dépendance
    # au point de fonctionnement de même signe.
    signes = [
        np.sign(e.r_residu_couple)
        for e in dynamiques
        if np.isfinite(e.r_residu_couple)
        and abs(e.r_residu_couple) > cfg.diagnostic.seuil_correlation_point_fct
    ]
    reproductible = None
    if len(dynamiques) >= 2:
        reproductible = len(signes) >= 2 and len(set(signes)) == 1

    reg_reference = next(
        (e.regression for e in campagne.par_type("balayage")
         if e.regression and not e.regression.non_calculable),
        None,
    )
    for e in dynamiques:
        e.diagnostic = M.diagnostiquer(
            nom_essai=f"l'essai « {e.nom} »",
            # Une régression propre à l'essai dynamique est peu robuste (le point
            # de fonctionnement bouge en permanence) : on s'appuie sur le
            # balayage pour l'offset et le gain, qui sont des caractéristiques du
            # capteur et non de l'essai.
            reg=e.regression if (e.regression and not e.regression.non_calculable) else reg_reference,
            recal=e.recalage_principal,
            correlation_residu_regime=e.r_residu_regime,
            correlation_residu_couple=e.r_residu_couple,
            pleine_echelle_Nm=cfg.pleine_echelle_Nm,
            params=cfg.diagnostic,
            reproductible=reproductible,
        )


def analyser(cfg: Config) -> ResultatCampagne:
    """Exécute la campagne complète et renvoie tous les résultats."""
    dossier_figures = cfg.dossier_sortie / "figures"
    dossier_figures.mkdir(parents=True, exist_ok=True)

    campagne = ResultatCampagne(
        config=cfg, avertissements=cfg.verifier_mapping() + cfg.verifier_saisies()
    )
    for essai in cfg.essais:
        campagne.essais.append(traiter_essai(essai, cfg, dossier_figures))

    # -- sensibilité thermique consolidée sur toute la campagne --------------
    T_all, res_all, couple_all = [], [], []
    for e in campagne.essais:
        if e.pool_temperature is not None:
            T_all.append(e.pool_temperature)
            res_all.append(e.pool_residu)
            couple_all.append(e.pool_couple)
    if T_all:
        campagne.thermique_globale = M.sensibilite_thermique(
            np.concatenate(T_all), np.concatenate(res_all),
            cfg.pleine_echelle_Nm, cfg.thermique, couple=np.concatenate(couple_all),
        )
        campagne.figures.append(
            graphiques.figure_residu_temperature(
                campagne.thermique_globale, cfg.pleine_echelle_Nm,
                dossier_figures / "residu_vs_temperature.png",
                titre="Campagne complète — résidu (mesuré − référence) vs température",
            )
        )
    else:
        campagne.thermique_globale = M.SensibiliteThermique(
            non_calculable="sensibilité thermique non calculable : aucun canal de température "
            "mappé sur les essais de la campagne"
        )

    # -- paramètres des cartes de contrôle, issus des essais répétés ---------
    # Les résidus des essais répétés sont centrés PAR NIVEAU de couple avant
    # d'alimenter la carte. Sans ce centrage, la série alternerait entre
    # niveaux de couple et l'estimateur robuste par étendue mobile, qui suppose
    # des observations consécutives du même point de fonctionnement, mesurerait
    # ces sauts de niveau au lieu de la répétabilité. La moyenne d'ensemble est
    # réinjectée pour que μ0 reste le résidu moyen réellement observé.
    residus_repetes = _residus_centres_par_niveau(campagne, cfg)
    # σ0 = répétabilité poolée intra-niveau (cf. `parametres_spc`) : c'est la
    # variabilité de la chaîne de mesure à couple constant, et non l'étalement
    # du résidu à travers la plage d'essai, qu'une erreur de gain dominerait.
    reps = [
        e.repetabilite
        for e in campagne.par_type("repetabilite")
        if e.repetabilite and not e.repetabilite.non_calculable
    ]
    ddl = sum(r.degres_liberte for r in reps)
    sigma0 = None
    if reps and ddl:
        sigma0 = math.sqrt(
            sum(r.degres_liberte * r.ecart_type_pc_pe**2 for r in reps) / ddl
        )
    campagne.spc = M.parametres_spc(residus_repetes, ddl=ddl, sigma0_pc_pe=sigma0)

    campagne.repetabilite_remontage, campagne.motif_remontage = _repetabilite_remontage(campagne, cfg)
    _diagnostics(campagne, cfg)
    campagne.incertitude = _incertitude(campagne, cfg)
    return campagne
