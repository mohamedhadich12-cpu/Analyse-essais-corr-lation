"""Configuration du pipeline de corrélation couple.

Toutes les hypothèses de traitement sont regroupées ici afin qu'aucune valeur
ne soit codée en dur dans les traitements, et qu'elles soient relisibles et
citables telles quelles dans le rapport.

Le fichier de configuration est un YAML ; voir `config/correlation.example.yaml`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

# Noms logiques manipulés par le pipeline. Le fichier de configuration fait la
# correspondance entre ces noms logiques et les libellés réels des canaux MDF4,
# qui diffèrent potentiellement d'un sous-dossier d'essai à l'autre.
CANAUX_SCALAIRES = (
    "couple_mesure_gauche",
    "couple_mesure_droite",
    # La référence banc peut être une voie unique (couple total ou couple d'une
    # transmission) ou deux voies gauche/droite. Les deux formes sont acceptées :
    # si les deux voies sont renseignées, elles priment sur la voie unique.
    "couple_reference",
    "couple_reference_gauche",
    "couple_reference_droite",
    "regime",
    "temperature",
)

TYPES_ESSAI = ("balayage", "repetabilite", "dynamique")

# Voie de couple mesuré confrontée à la référence banc. Le choix est explicite
# et jamais deviné : avec deux transmissions instrumentées et une référence
# banc au même point de la chaîne (rapport 1:1), « moyenne » est le choix
# cohérent sur un essai en ligne droite ; « somme » convient si la référence
# représente le couple total aux roues.
MODES_COMPARAISON = ("gauche", "droite", "moyenne", "somme")


class ErreurConfig(RuntimeError):
    """Configuration invalide ou incomplète.

    Levée avec un message explicite plutôt que de deviner une valeur par
    défaut : une hypothèse silencieuse sur un mapping de canal fausserait
    l'ensemble des résultats métrologiques.
    """


def _construire(cls, donnees: Any, contexte: str):
    """Instancie une dataclass de paramètres depuis un dict YAML.

    Toute clé inconnue lève une erreur : cela évite qu'une faute de frappe dans
    le YAML passe inaperçue et laisse silencieusement la valeur par défaut.
    """
    if donnees is None:
        return cls()
    if not isinstance(donnees, dict):
        raise ErreurConfig(f"{contexte} : un dictionnaire est attendu, reçu {type(donnees).__name__}.")
    connues = {f.name for f in fields(cls)}
    inconnues = set(donnees) - connues
    if inconnues:
        raise ErreurConfig(
            f"{contexte} : clé(s) inconnue(s) {sorted(inconnues)}. "
            f"Clés acceptées : {sorted(connues)}."
        )
    return cls(**donnees)


@dataclass(frozen=True)
class MappingCanaux:
    """Libellés réels des canaux MDF4 pour un essai donné.

    `None` signifie « canal absent de cette acquisition » : les grandeurs qui
    en dépendent seront déclarées non calculables, pas estimées.
    """

    couple_mesure_gauche: str | None = None
    couple_mesure_droite: str | None = None
    couple_reference: str | None = None
    couple_reference_gauche: str | None = None
    couple_reference_droite: str | None = None
    regime: str | None = None
    temperature: str | None = None
    # Canaux d'état (LED télémétrie, CRC, compteurs d'erreurs...). Simplement
    # relevés et résumés : la valeur « nominale » n'étant pas connue a priori,
    # le pipeline restitue la distribution des valeurs sans l'interpréter.
    etat: tuple[str, ...] = ()

    @property
    def voies_couple_mesure(self) -> tuple[str, ...]:
        return tuple(
            n for n in (self.couple_mesure_gauche, self.couple_mesure_droite) if n
        )

    @property
    def voies_couple_reference(self) -> tuple[str, ...]:
        return tuple(
            n for n in (self.couple_reference_gauche, self.couple_reference_droite) if n
        )

    @property
    def a_reference(self) -> bool:
        """Vrai dès qu'au moins une voie de référence est renseignée."""
        return bool(self.couple_reference or self.voies_couple_reference)

    def presents(self) -> dict[str, str | None]:
        return {n: getattr(self, n) for n in CANAUX_SCALAIRES}


@dataclass(frozen=True)
class ParamsPaliers:
    """Détection des points stabilisés (essais de balayage / répétabilité).

    HYPOTHÈSES (documentées dans le rapport) :
      * un palier est une fenêtre glissante de `duree_palier_s` secondes sur
        laquelle l'écart-type du couple de référence reste sous
        `tolerance_stab_pc_pe` % de la pleine échelle ;
      * la valeur retenue pour le palier est la moyenne sur la
        `fraction_finale` dernière partie du palier, afin d'écarter le
        transitoire d'établissement ;
      * deux paliers consécutifs sont considérés distincts si leurs niveaux de
        référence diffèrent de plus de `ecart_min_paliers_pc_pe` % PE.
    """

    duree_palier_s: float = 2.0
    tolerance_stab_pc_pe: float = 0.5
    fraction_finale: float = 0.5
    ecart_min_paliers_pc_pe: float = 1.0
    # Appariement montée/descente pour l'hystérésis, et regroupement des
    # répétitions pour la répétabilité.
    tolerance_appariement_pc_pe: float = 1.0


@dataclass(frozen=True)
class ParamsIntercorrelation:
    """Recalage temporel par intercorrélation (essais dynamiques).

    HYPOTHÈSES :
      * le retard recherché est borné à ±`retard_max_ms` ; au-delà, le pic
        d'intercorrélation serait un repliement sur une autre sollicitation ;
      * les deux signaux sont filtrés passe-haut à `passe_haut_Hz` avant
        corrélation : sans cela l'offset et les composantes lentes dominent le
        produit de corrélation et aplatissent le pic ;
      * seules les portions réellement dynamiques sont corrélées, définies par
        un écart-type glissant du couple de référence supérieur à
        `seuil_activite_pc_pe` % PE ;
      * les interruptions de moins de `duree_comblement_s` dans cette détection
        sont comblées : un cycle transitoire passe par des extrema où la
        variance instantanée s'annule sans que la phase cesse d'être dynamique,
        et sans ce comblement la fenêtre d'analyse serait fragmentée en
        tronçons trop courts pour identifier le retard.
    """

    retard_max_ms: float = 500.0
    passe_haut_Hz: float | None = 0.2
    seuil_activite_pc_pe: float = 2.0
    fenetre_activite_s: float = 1.0
    duree_comblement_s: float = 5.0


@dataclass(frozen=True)
class ParamsZero:
    """Détection des relevés de zéro en début et fin d'essai.

    HYPOTHÈSES : un relevé de zéro est une fenêtre d'au moins
    `duree_fenetre_s` secondes pendant laquelle le couple de référence est sous
    `seuil_couple_ref_pc_pe` % PE en valeur absolue et, si le canal existe, le
    régime est sous `seuil_regime` (unité du canal régime).
    """

    duree_fenetre_s: float = 5.0
    seuil_couple_ref_pc_pe: float = 1.0
    seuil_regime: float | None = 20.0
    # La plage « début » doit commencer dans cette fraction initiale de l'essai
    # et la plage « fin » se terminer dans la fraction finale correspondante ;
    # sinon il s'agit d'un passage à couple nul en cours d'essai et non d'un
    # relevé de zéro avant/après.
    fraction_bord: float = 0.25


@dataclass(frozen=True)
class ParamsThermique:
    """Régression du résidu sur la température.

    HYPOTHÈSE : en deçà de `amplitude_min_C` d'excursion thermique sur l'essai,
    la pente est jugée non identifiable et la grandeur est déclarée non
    calculable plutôt qu'extrapolée.
    """

    amplitude_min_C: float = 5.0
    # Plage de température rencontrée en service, utilisée UNIQUEMENT pour
    # convertir la sensibilité thermique en contribution au bilan
    # d'incertitude. Laisser `null` si elle n'est pas définie : la
    # contribution est alors exclue et l'incertitude annoncée est un minorant.
    plage_service_C: float | None = None


@dataclass(frozen=True)
class ParamsDiagnostic:
    """Seuils de décision pour la grille de conclusion qualitative.

    Ces seuils ne servent qu'à *rédiger* la conclusion (offset / gain /
    synchronisation / écart de modèle) ; ils n'entrent dans aucun calcul.
    """

    seuil_offset_pc_pe: float = 0.5
    seuil_gain_pc: float = 1.0
    seuil_retard_ms: float = 5.0
    # Baisse relative du RMS de résidu après recalage au-delà de laquelle
    # l'écart est attribué à un défaut de synchronisation.
    gain_rms_recalage: float = 0.30
    # |r| de corrélation résidu / point de fonctionnement au-delà duquel on
    # conclut à une dépendance au point de fonctionnement.
    seuil_correlation_point_fct: float = 0.5


@dataclass(frozen=True)
class DeclarationEssai:
    """Un sous-dossier d'essai et le traitement qui lui est appliqué."""

    dossier: str
    type: str
    # Essai en ligne droite sans sollicitation différentielle : autorise
    # l'exploitation du résidu gauche − droite comme indicateur de redondance.
    ligne_droite: bool = False
    # Étiquette de groupe pour la répétabilité après remontage (ex. "avant" /
    # "apres"). Sans au moins deux groupes distincts renseignés, la
    # répétabilité APRÈS REMONTAGE est déclarée non calculable.
    groupe_remontage: str | None = None
    # Limite le nombre de fichiers traités (mise au point). None = tous.
    max_fichiers: int | None = None

    def __post_init__(self) -> None:
        if self.type not in TYPES_ESSAI:
            raise ErreurConfig(
                f"Essai « {self.dossier} » : type « {self.type} » inconnu. "
                f"Types acceptés : {list(TYPES_ESSAI)}."
            )


@dataclass
class Config:
    racine: Path
    dossier_sortie: Path
    pleine_echelle_Nm: float
    mode_comparaison: str
    rapport_reduction: float
    incertitude_reference_k1_Nm: float | None
    frequence_Hz: float
    # Un démontage/remontage a-t-il été réalisé au cours de la campagne ?
    # `False` distingue, dans le rapport, une grandeur **non définie** (aucun
    # remontage effectué, il n'y a rien à mesurer) d'une grandeur simplement
    # **non configurée** (remontage effectué mais groupes non déclarés) — les
    # deux mènent à « non calculable », mais ne se justifient pas pareil.
    # `None` = non précisé.
    remontage_realise: bool | None
    essais: tuple[DeclarationEssai, ...]
    _canaux_defaut: dict[str, Any]
    _canaux_par_dossier: dict[str, dict[str, Any]]
    paliers: ParamsPaliers = field(default_factory=ParamsPaliers)
    intercorrelation: ParamsIntercorrelation = field(default_factory=ParamsIntercorrelation)
    zero: ParamsZero = field(default_factory=ParamsZero)
    thermique: ParamsThermique = field(default_factory=ParamsThermique)
    diagnostic: ParamsDiagnostic = field(default_factory=ParamsDiagnostic)

    # -- accès ----------------------------------------------------------
    def canaux_pour(self, dossier: str) -> MappingCanaux:
        """Mapping effectif d'un sous-dossier : défaut surchargé par dossier."""
        fusion = dict(self._canaux_defaut)
        fusion.update(self._canaux_par_dossier.get(dossier, {}))
        etat = fusion.pop("etat", ()) or ()
        if isinstance(etat, str):
            etat = (etat,)
        inconnues = set(fusion) - set(CANAUX_SCALAIRES)
        if inconnues:
            raise ErreurConfig(
                f"canaux (dossier « {dossier} ») : clé(s) inconnue(s) {sorted(inconnues)}. "
                f"Clés acceptées : {sorted(CANAUX_SCALAIRES)} + 'etat'."
            )
        return MappingCanaux(**fusion, etat=tuple(etat))

    def chemin_essai(self, dossier: str) -> Path:
        return self.racine / dossier

    def pc_pe(self, valeur_Nm: float) -> float:
        """Convertit des N·m en % de la pleine échelle."""
        return 100.0 * valeur_Nm / self.pleine_echelle_Nm

    def nm(self, valeur_pc_pe: float) -> float:
        """Convertit des % de pleine échelle en N·m."""
        return valeur_pc_pe * self.pleine_echelle_Nm / 100.0

    # -- chargement -----------------------------------------------------
    @classmethod
    def charger(cls, chemin: str | Path) -> "Config":
        chemin = Path(chemin)
        if not chemin.is_file():
            raise ErreurConfig(f"Fichier de configuration introuvable : {chemin}")
        with open(chemin, "r", encoding="utf-8") as fh:
            brut = yaml.safe_load(fh) or {}
        return cls.depuis_dict(brut, source=chemin)

    @classmethod
    def depuis_dict(cls, brut: dict[str, Any], source: Path | None = None) -> "Config":
        ctx = f" ({source})" if source else ""

        def requis(cle: str):
            if cle not in brut or brut[cle] is None:
                raise ErreurConfig(f"Clé « {cle} » manquante dans la configuration{ctx}.")
            return brut[cle]

        comparaison = brut.get("comparaison") or {}
        mode = comparaison.get("mode", "moyenne")
        if mode not in MODES_COMPARAISON:
            raise ErreurConfig(
                f"comparaison.mode = « {mode} » inconnu. Valeurs acceptées : {list(MODES_COMPARAISON)}."
            )

        acquisition = brut.get("acquisition") or {}
        frequence = float(acquisition.get("frequence_reechantillonnage_Hz", 100.0))
        if frequence <= 0:
            raise ErreurConfig("acquisition.frequence_reechantillonnage_Hz doit être > 0.")

        canaux = brut.get("canaux") or {}
        defaut = canaux.get("defaut") or {}
        par_dossier = canaux.get("par_dossier") or {}
        if not isinstance(par_dossier, dict):
            raise ErreurConfig("canaux.par_dossier doit être un dictionnaire dossier -> mapping.")
        par_dossier = {k: (v or {}) for k, v in par_dossier.items()}

        essais_bruts = requis("essais")
        if not isinstance(essais_bruts, dict):
            raise ErreurConfig("essais doit être un dictionnaire dossier -> {type: ...}.")
        essais = []
        for dossier, spec in essais_bruts.items():
            spec = dict(spec or {})
            if "type" not in spec:
                raise ErreurConfig(f"Essai « {dossier} » : clé « type » obligatoire ({list(TYPES_ESSAI)}).")
            connues = {f.name for f in fields(DeclarationEssai)} - {"dossier"}
            inconnues = set(spec) - connues
            if inconnues:
                raise ErreurConfig(
                    f"Essai « {dossier} » : clé(s) inconnue(s) {sorted(inconnues)}. "
                    f"Clés acceptées : {sorted(connues)}."
                )
            essais.append(DeclarationEssai(dossier=dossier, **spec))

        pe = float(requis("pleine_echelle_Nm"))
        if pe <= 0:
            raise ErreurConfig("pleine_echelle_Nm doit être > 0.")

        u_ref = comparaison.get("incertitude_reference_k1_Nm")

        remontage = brut.get("remontage") or {}
        if not isinstance(remontage, dict):
            raise ErreurConfig("remontage doit être un dictionnaire (clé `realise`).")
        inconnues = set(remontage) - {"realise"}
        if inconnues:
            raise ErreurConfig(
                f"remontage : clé(s) inconnue(s) {sorted(inconnues)}. Clé acceptée : ['realise']."
            )
        realise = remontage.get("realise")

        return cls(
            racine=Path(str(requis("racine_donnees"))).expanduser(),
            dossier_sortie=Path(str(brut.get("dossier_sortie", "sortie"))).expanduser(),
            pleine_echelle_Nm=pe,
            mode_comparaison=mode,
            rapport_reduction=float(comparaison.get("rapport_reduction", 1.0)),
            incertitude_reference_k1_Nm=None if u_ref is None else float(u_ref),
            frequence_Hz=frequence,
            remontage_realise=None if realise is None else bool(realise),
            essais=tuple(essais),
            _canaux_defaut=dict(defaut),
            _canaux_par_dossier=par_dossier,
            paliers=_construire(ParamsPaliers, brut.get("paliers"), "paliers"),
            intercorrelation=_construire(
                ParamsIntercorrelation, brut.get("intercorrelation"), "intercorrelation"
            ),
            zero=_construire(ParamsZero, brut.get("zero"), "zero"),
            thermique=_construire(ParamsThermique, brut.get("thermique"), "thermique"),
            diagnostic=_construire(ParamsDiagnostic, brut.get("diagnostic"), "diagnostic"),
        )

    def verifier_saisies(self) -> list[str]:
        """Signale les valeurs qui relèvent d'une saisie utilisateur et manquent.

        Ces grandeurs caractérisent le moyen d'essai ou le domaine d'emploi :
        aucun traitement du signal ne peut les produire. Leur absence n'est pas
        une erreur — le bilan reste calculé, annoncé comme minorant — mais elle
        doit être visible dès l'exécution, et pas seulement à la relecture du
        rapport.
        """
        avertissements: list[str] = []
        if self.incertitude_reference_k1_Nm is None:
            avertissements.append(
                "`comparaison.incertitude_reference_k1_Nm` non renseignée (saisie utilisateur, "
                "à reprendre du certificat d'étalonnage du banc GMP ; si le certificat donne "
                "une incertitude élargie à k=2, saisir la moitié) → contribution exclue, "
                "l'incertitude élargie sera un MINORANT."
            )
        if self.thermique.plage_service_C is None:
            avertissements.append(
                "`thermique.plage_service_C` non renseignée (saisie utilisateur : domaine "
                "d'emploi, que l'excursion constatée sur les essais ne peut pas remplacer) "
                "→ contribution thermique exclue du bilan d'incertitude."
            )
        return avertissements

    def verifier_mapping(self) -> list[str]:
        """Retourne la liste des avertissements de mapping (canaux non renseignés).

        N'échoue pas : un canal absent est une information, pas une erreur — il
        rendra simplement certaines grandeurs non calculables.
        """
        avertissements: list[str] = []
        for essai in self.essais:
            m = self.canaux_pour(essai.dossier)
            if not m.a_reference:
                avertissements.append(
                    f"« {essai.dossier} » : aucune voie de couple de référence renseignée "
                    "→ aucune comparaison à la référence banc possible sur cet essai."
                )
            elif len(m.voies_couple_mesure) == 2 and len(m.voies_couple_reference) < 2:
                # Comparer une moyenne de deux voies mesurées à une référence unique
                # n'a de sens que si l'on sait ce que cette référence représente.
                avertissements.append(
                    f"« {essai.dossier} » : couple mesuré sur 2 voies mais référence sur une "
                    f"seule → vérifier la cohérence du mode « {self.mode_comparaison} » "
                    "(cette référence est-elle le couple total ou celui d'une seule transmission ?)."
                )
            if not m.voies_couple_mesure:
                avertissements.append(
                    f"« {essai.dossier} » : aucune voie de couple mesuré renseignée "
                    "→ essai ignoré."
                )
            if essai.ligne_droite and len(m.voies_couple_mesure) < 2:
                avertissements.append(
                    f"« {essai.dossier} » : déclaré en ligne droite mais une seule voie de "
                    "couple mappée → résidu gauche−droite non calculable."
                )
            if not m.temperature:
                avertissements.append(
                    f"« {essai.dossier} » : pas de canal température → sensibilité thermique "
                    "non calculable sur cet essai."
                )
        return avertissements
