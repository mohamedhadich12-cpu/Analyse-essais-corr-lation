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
from . import zones as Z
from .config import Config, DeclarationEssai
from .io_mdf import ErreurChargement, SignauxEssai, resumer_etats
from .lecteurs import charger_signaux, doublons_de_format, lister_fichiers

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


# Au-delà, le tracé s'alourdit sans rien montrer de plus : deux points ne
# peuvent pas occuper le même pixel. Ne concerne que l'AFFICHAGE — les calculs
# portent toujours sur la totalité des échantillons.
POINTS_TRACES_MAX = 6000


def _decimation(t: np.ndarray) -> int:
    return max(1, int(np.ceil(t.size / POINTS_TRACES_MAX)))


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
    avertissements: list[str] = field(default_factory=list)
    paliers: list[M.Palier] = field(default_factory=list)
    regression: M.Regression | None = None
    hysteresis: M.Hysteresis | None = None
    non_linearite_Nm: float = float("nan")
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
    # Résultats consolidés, quel que soit le mode d'organisation. Le rapport ne
    # lit qu'eux : il n'a pas à savoir si les essais ont été déclarés par
    # l'utilisateur ou si leurs zones ont été découvertes automatiquement.
    mode: str = "declare"
    zones: list = field(default_factory=list, repr=False)
    regression_globale: M.Regression | None = None
    source_regression: str = ""
    hysteresis_globale: M.Hysteresis | None = None
    non_linearite_Nm: float = float("nan")
    repetabilite_globale: M.Repetabilite | None = None
    recalages_globaux: list[tuple[Path, M.Recalage]] = field(default_factory=list, repr=False)
    derives_zero_globales: list[tuple[str, M.DeriveZero]] = field(default_factory=list, repr=False)
    thermique_globale: M.SensibiliteThermique | None = None
    spc: M.ParametresSPC | None = None
    repetabilite_remontage: M.Repetabilite | None = None
    motif_remontage: str | None = None
    redondance: M.RedondanceGD | None = None
    incertitude: M.BilanIncertitude | None = None
    avertissements: list[str] = field(default_factory=list)
    figures: list[Path] = field(default_factory=list)
    # Acquisitions retenues pour les deux figures qui portent sur UN fichier.
    # Les garder permet à l'interface de retracer exactement la même, plutôt
    # que d'en choisir une autre et de contredire l'image du rapport.
    fichier_recalage: Path | None = None
    fichier_redondance: Path | None = None

    def par_type(self, type_essai: str) -> list[ResultatEssai]:
        return [e for e in self.essais if e.declaration.type == type_essai]

    @property
    def retards_ms(self) -> list[float]:
        return [r.retard_ms for _, r in self.recalages_globaux if not r.non_calculable]

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
        resultat.erreurs.append(f"aucune acquisition exploitable trouvée dans {chemin}")
        return resultat

    for racine, doublons in doublons_de_format(fichiers).items():
        resultat.avertissements.append(
            f"« {essai.dossier} » : l'acquisition « {racine} » est présente sous "
            f"{len(doublons)} formats ({', '.join(f.suffix for f in doublons)}). "
            "Ces fichiers portant les mêmes grandeurs, ils seront traités comme des "
            "essais distincts et compteront double dans tout ce qui se cumule "
            "(paliers, répétabilité, dispersion des cartes). Ne conservez qu'un "
            "format par acquisition."
        )
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
        resultat.non_linearite_Nm = M.non_linearite_Nm(reg, cfg.pleine_echelle_Nm)
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
    tolerance = cfg.paliers.tolerance_appariement_Nm
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
    """Résidus des essais répétés (N·m), centrés par essai ET par niveau de couple.

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
    tolerance = cfg.paliers.tolerance_appariement_Nm
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
                p.residu - moyenne_groupe + moyenne_globale
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

    if reg_essai and np.isfinite(reg_essai.non_linearite_Nm):
        contributions.append(
            M.Contribution(
                "Non-linéarité", reg_essai.non_linearite_Nm, "rectangulaire",
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
            M.Contribution("Hystérésis", hyst.max_Nm / 2.0, "rectangulaire",
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

    derives = [d.derive_Nm for e in campagne.essais for d in e.derives_zero_valides]
    if derives:
        pire = max(derives, key=abs)
        contributions.append(
            M.Contribution("Dérive de zéro", pire, "rectangulaire",
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


def analyser_declare(cfg: Config) -> ResultatCampagne:
    """Exécute la campagne à partir des essais DÉCLARÉS en configuration."""
    dossier_figures = cfg.dossier_sortie / "figures"
    dossier_figures.mkdir(parents=True, exist_ok=True)

    campagne = ResultatCampagne(
        config=cfg, avertissements=cfg.verifier_mapping() + cfg.verifier_saisies()
    )
    for essai in cfg.essais:
        resultat = traiter_essai(essai, cfg, dossier_figures)
        campagne.essais.append(resultat)
        # Les avertissements des essais remontent au niveau campagne : ils
        # doivent apparaître dans le rapport et à l'écran, pas seulement dans le
        # détail d'un essai que personne ne déplie.
        campagne.avertissements.extend(resultat.avertissements)

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
            sum(r.degres_liberte * r.ecart_type_Nm**2 for r in reps) / ddl
        )
    campagne.spc = M.parametres_spc(residus_repetes, ddl=ddl, sigma0_Nm=sigma0)

    campagne.repetabilite_remontage, campagne.motif_remontage = _repetabilite_remontage(campagne, cfg)
    _diagnostics(campagne, cfg)
    campagne.incertitude = _incertitude(campagne, cfg)
    _consolider_depuis_essais(campagne, cfg)
    return campagne


# ---------------------------------------------------------------------------
# Mode automatique : un dossier d'acquisitions, sans déclaration de types
# ---------------------------------------------------------------------------


def _consolider_depuis_essais(campagne: ResultatCampagne, cfg: Config) -> None:
    """Remonte au niveau campagne les résultats des essais déclarés.

    Le rapport ne lit que les champs consolidés : il n'a pas à savoir comment la
    campagne a été organisée.
    """
    balayages = campagne.par_type("balayage")
    essai = next((e for e in balayages if e.regression and not e.regression.non_calculable), None)
    if essai:
        campagne.regression_globale = essai.regression
        campagne.source_regression = f"essai « {essai.nom} »"
        campagne.non_linearite_Nm = essai.non_linearite_Nm
    elif balayages:
        campagne.regression_globale = balayages[0].regression
    campagne.hysteresis_globale = next(
        (e.hysteresis for e in balayages if e.hysteresis and not e.hysteresis.non_calculable),
        balayages[0].hysteresis if balayages else None,
    )
    campagne.repetabilite_globale = next(
        (e.repetabilite for e in campagne.par_type("repetabilite")
         if e.repetabilite and not e.repetabilite.non_calculable),
        None,
    )
    campagne.recalages_globaux = [
        (Path(e.nom), e.recalage_principal)
        for e in campagne.par_type("dynamique")
        if e.recalage_principal is not None
    ]
    campagne.redondance = next(
        (e.redondance for e in campagne.essais
         if e.redondance and not e.redondance.non_calculable),
        None,
    )
    campagne.derives_zero_globales = [
        (e.nom, d) for e in campagne.essais for d in e.derives_zero_valides
    ]


def analyser_automatique(cfg: Config) -> ResultatCampagne:
    """Analyse un dossier d'acquisitions **sans déclaration de types d'essai**.

    Tous les fichiers exploitables sous la racine sont lus, quelle que soit leur
    place dans l'arborescence. Chacun est examiné pour les zones qu'il contient
    (cf. `zones`), et chaque grandeur est calculée à partir de la mise en commun
    des zones qui la concernent :

      * **régression, non-linéarité** : tous les paliers de tous les fichiers ;
      * **hystérésis** : les fichiers qui présentent montée ET descente ;
      * **répétabilité** : les niveaux de couple atteints par au moins deux
        fichiers différents ;
      * **retard** : les fichiers présentant une plage dynamique identifiable ;
      * **dérive de zéro** : les fichiers ayant un repos au début et à la fin ;
      * **thermique** : tous les échantillons disposant d'une température.

    Ce que la détection ne peut pas deviner reste déclaratif : la symétrie de
    chargement du banc, et l'existence d'un démontage/remontage.
    """
    dossier_figures = cfg.dossier_sortie / "figures"
    dossier_figures.mkdir(parents=True, exist_ok=True)

    campagne = ResultatCampagne(
        config=cfg, mode="automatique",
        avertissements=cfg.verifier_saisies(),
    )
    mapping = cfg.canaux_pour("")
    if not mapping.a_reference or not mapping.voies_couple_mesure:
        campagne.avertissements.append(
            "Mapping incomplet : sans voie de couple mesuré ET de couple de référence, "
            "aucune grandeur n'est calculable."
        )
        return campagne

    fichiers = lister_fichiers(cfg.racine)
    if not fichiers:
        campagne.avertissements.append(f"Aucune acquisition exploitable sous {cfg.racine}.")
        return campagne
    for racine_commune, doublons in doublons_de_format(fichiers).items():
        campagne.avertissements.append(
            f"L'acquisition « {racine_commune} » est présente sous {len(doublons)} formats "
            f"({', '.join(f.suffix for f in doublons)}) : elle compterait double. "
            "Ne conservez qu'un format par acquisition."
        )

    T_pool, res_pool, couple_pool = [], [], []
    paliers_tous: list[M.Palier] = []
    donnees_figure = None
    meilleur_pic = -np.inf

    for fichier in fichiers:
        try:
            donnees = preparer(fichier, cfg, mapping)
        except Exception as exc:
            campagne.zones.append(Z.ZonesFichier(chemin=fichier, duree_s=0.0, erreur=str(exc)))
            continue

        zones = Z.detecter(donnees, cfg)
        campagne.zones.append(zones)
        paliers_tous.extend(zones.paliers)

        # Le tableau des zones dit combien ; cette figure dit où. Sans elle, un
        # palier mal placé ou une plage dynamique qui déborde sur un arrêt ne se
        # verraient pas — la détection resterait à croire sur parole.
        campagne.figures.append(
            graphiques.figure_zones(
                donnees.t, donnees.reference, donnees.mesure, zones,
                cfg.pleine_echelle_Nm,
                dossier_figures / f"zones_{slug(fichier.stem)}.png",
                titre=f"{fichier.name} — zones détectées",
                decimation=_decimation(donnees.t),
            )
        )

        residu_thermique = donnees.residu
        if zones.alimente_retard:
            recal = M.recalage_temporel(
                donnees.t, donnees.reference, donnees.mesure,
                cfg.pleine_echelle_Nm, cfg.intercorrelation,
            )
            campagne.recalages_globaux.append((fichier, recal))
            if not recal.non_calculable:
                residu_thermique = (
                    M.appliquer_retard(donnees.t, donnees.mesure, recal.retard_ms / 1000.0)
                    - donnees.reference
                )
                if recal.correlation_pic > meilleur_pic:
                    meilleur_pic, donnees_figure = recal.correlation_pic, (donnees, recal)

        if zones.alimente_derive_zero:
            derive = M.derive_zero(
                donnees.t, donnees.mesure, donnees.reference, donnees.regime,
                cfg.pleine_echelle_Nm, cfg.zero,
            )
            if not derive.non_calculable:
                campagne.derives_zero_globales.append((fichier.name, derive))

        if donnees.temperature is not None:
            pas = max(1, int(round(cfg.frequence_Hz / FREQUENCE_THERMIQUE_Hz)))
            T_pool.append(donnees.temperature[::pas])
            res_pool.append(residu_thermique[::pas])
            couple_pool.append(donnees.reference[::pas])

        if cfg.ligne_droite and campagne.redondance is None:
            campagne.redondance = M.redondance_gauche_droite(
                donnees.gauche, donnees.droite, cfg.pleine_echelle_Nm
            )
            if not campagne.redondance.non_calculable:
                campagne.fichier_redondance = fichier
                campagne.figures.append(
                    graphiques.figure_redondance(
                        donnees.t, donnees.gauche, donnees.droite, cfg.pleine_echelle_Nm,
                        dossier_figures / "redondance_gd.png",
                        titre=f"{fichier.name} — redondance des voies (gauche − droite)",
                    )
                )

    # -- grandeurs issues des paliers, tous fichiers confondus --------------
    if paliers_tous:
        campagne.regression_globale = M.regression(
            [p.reference for p in paliers_tous], [p.mesure for p in paliers_tous]
        )
        campagne.source_regression = (
            f"{len(paliers_tous)} paliers de "
            f"{len({p.source for p in paliers_tous})} acquisition(s)"
        )
        campagne.non_linearite_Nm = M.non_linearite_Nm(
            campagne.regression_globale, cfg.pleine_echelle_Nm
        )
        campagne.figures.append(
            graphiques.figure_regression_balayage(
                paliers_tous, campagne.regression_globale,
                M.hysteresis(paliers_tous, cfg.pleine_echelle_Nm, cfg.paliers),
                cfg.pleine_echelle_Nm, dossier_figures / "regression_paliers.png",
                titre="Paliers stabilisés — corrélation transmissions / banc GMP",
            )
        )

    # L'hystérésis n'a de sens qu'au sein d'UN fichier : apparier une montée
    # d'une acquisition à une descente d'une autre confondrait l'hystérésis avec
    # la dispersion entre essais.
    hysteresis_par_fichier = [
        M.hysteresis(z.paliers, cfg.pleine_echelle_Nm, cfg.paliers)
        for z in campagne.zones if z.alimente_hysteresis
    ]
    valides = [h for h in hysteresis_par_fichier if not h.non_calculable]
    if valides:
        campagne.hysteresis_globale = max(valides, key=lambda h: h.max_Nm)
    elif hysteresis_par_fichier:
        campagne.hysteresis_globale = hysteresis_par_fichier[0]
    else:
        campagne.hysteresis_globale = M.Hysteresis(
            non_calculable="hystérésis non calculable : aucune acquisition ne présente à la "
            "fois des paliers en montée et en descente"
        )

    # -- répétabilité : niveaux atteints par plusieurs fichiers -------------
    repetes = Z.niveaux_repetes(campagne.zones, cfg)
    paliers_repetes = [p for groupe in repetes.values() for p in groupe]
    if paliers_repetes:
        campagne.repetabilite_globale = M.repetabilite(
            paliers_repetes, cfg.pleine_echelle_Nm, cfg.paliers
        )
        campagne.figures.append(
            graphiques.figure_repetabilite(
                campagne.repetabilite_globale, cfg.pleine_echelle_Nm,
                dossier_figures / "repetabilite.png",
                titre="Répétabilité — niveaux atteints par plusieurs acquisitions",
            )
        )
    else:
        campagne.repetabilite_globale = M.Repetabilite(
            non_calculable="répétabilité non calculable : aucun niveau de couple n'est atteint "
            "par au moins deux acquisitions différentes"
        )

    # -- recalage : figure sur l'acquisition au pic le plus marqué ----------
    if donnees_figure is not None:
        donnees, recal = donnees_figure
        campagne.fichier_recalage = donnees.chemin
        campagne.figures.append(
            graphiques.figure_recalage(
                donnees.t, donnees.reference, donnees.mesure, recal,
                dossier_figures / "recalage.png", titre=donnees.chemin.name,
            )
        )

    # -- thermique ----------------------------------------------------------
    if T_pool:
        campagne.thermique_globale = M.sensibilite_thermique(
            np.concatenate(T_pool), np.concatenate(res_pool),
            cfg.pleine_echelle_Nm, cfg.thermique, couple=np.concatenate(couple_pool),
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
            "mappé"
        )

    # -- cartes de contrôle -------------------------------------------------
    campagne.spc = _spc_automatique(campagne, cfg, repetes)
    campagne.repetabilite_remontage = None
    campagne.motif_remontage = (
        "répétabilité après remontage non calculable : aucun démontage ni remontage de la "
        "chaîne de mesure n'a été réalisé au cours de la campagne. La grandeur n'est pas "
        "définie ici — elle n'est pas manquante"
        if cfg.remontage_realise is False
        else "répétabilité après remontage non calculable : en organisation automatique, "
        "les groupes avant/après remontage ne peuvent pas être devinés — déclarez-les en "
        "mode par dossier si la campagne en comporte"
    )
    campagne.incertitude = _incertitude_globale(campagne, cfg)
    return campagne


def _spc_automatique(campagne: ResultatCampagne, cfg: Config, repetes: dict) -> M.ParametresSPC:
    """μ0 et σ0 depuis les niveaux réellement répétés entre acquisitions."""
    if not repetes:
        return M.ParametresSPC(
            non_calculable="paramètres SPC non calculables : aucun niveau de couple n'est "
            "atteint par au moins deux acquisitions"
        )
    tous = [p for groupe in repetes.values() for p in groupe]
    moyenne_globale = float(np.mean([p.residu for p in tous]))
    centres = [
        p.residu - float(np.mean([q.residu for q in groupe])) + moyenne_globale
        for groupe in repetes.values()
        for p in groupe
    ]
    rep = campagne.repetabilite_globale
    sigma0 = rep.ecart_type_Nm if rep and not rep.non_calculable else None
    ddl = rep.degres_liberte if rep and not rep.non_calculable else 0
    return M.parametres_spc(centres, ddl=ddl, sigma0_Nm=sigma0)


def _incertitude_globale(campagne: ResultatCampagne, cfg: Config) -> M.BilanIncertitude:
    """Bilan d'incertitude bâti sur les grandeurs consolidées."""
    contributions: list[M.Contribution] = []
    exclusions: list[str] = []

    if math.isfinite(campagne.non_linearite_Nm):
        contributions.append(M.Contribution(
            "Non-linéarité", campagne.non_linearite_Nm, "rectangulaire",
            f"résidu max de la régression sur {campagne.source_regression}"))
    else:
        exclusions.append("non-linéarité (aucune régression exploitable)")

    hyst = campagne.hysteresis_globale
    if hyst and not hyst.non_calculable:
        contributions.append(M.Contribution(
            "Hystérésis", hyst.max_Nm / 2.0, "rectangulaire",
            "demi-écart montée/descente"))
    else:
        exclusions.append("hystérésis (aucun appariement montée/descente)")

    rep = campagne.repetabilite_globale
    if rep and not rep.non_calculable:
        contributions.append(M.Contribution(
            "Répétabilité", rep.ecart_type_Nm, "normale",
            f"écart-type poolé, {rep.degres_liberte} ddl"))
    else:
        exclusions.append("répétabilité (aucun niveau répété)")

    derives = [d.derive_Nm for _, d in campagne.derives_zero_globales]
    if derives:
        contributions.append(M.Contribution(
            "Dérive de zéro", max(derives, key=abs), "rectangulaire",
            f"dérive la plus forte sur {len(derives)} acquisition(s)"))
    else:
        exclusions.append("dérive de zéro (aucun relevé de zéro avant/après identifiable)")

    th = campagne.thermique_globale
    if th and not th.non_calculable and cfg.thermique.plage_service_C:
        contributions.append(M.Contribution(
            "Effet thermique",
            abs(th.pente_Nm_par_C) * cfg.thermique.plage_service_C / 2.0, "rectangulaire",
            f"sur une plage de service de {cfg.thermique.plage_service_C:.0f} °C"))
    elif th and not th.non_calculable:
        exclusions.append("effet thermique (`thermique.plage_service_C` non renseignée)")
    else:
        exclusions.append("effet thermique (sensibilité thermique non calculable)")

    if cfg.incertitude_reference_k1_Nm is not None:
        contributions.append(M.Contribution(
            "Référence banc GMP", cfg.incertitude_reference_k1_Nm, "normale",
            "incertitude-type du moyen de référence, fournie en configuration"))
    else:
        exclusions.append(
            "incertitude du moyen de référence (`comparaison.incertitude_reference_k1_Nm` "
            "non renseignée : elle ne peut pas être déduite des acquisitions)")

    return M.bilan_incertitude(contributions, exclusions, cfg.pleine_echelle_Nm)


def analyser(cfg: Config) -> ResultatCampagne:
    """Point d'entrée unique : choisit l'organisation d'après la configuration.

    Sans essai déclaré, les acquisitions sont parcourues à plat et leurs zones
    découvertes automatiquement. Avec des essais déclarés, chaque dossier reçoit
    le traitement de son type — plus précis lorsqu'on connaît le protocole.
    """
    if cfg.organisation_automatique:
        return analyser_automatique(cfg)
    return analyser_declare(cfg)
