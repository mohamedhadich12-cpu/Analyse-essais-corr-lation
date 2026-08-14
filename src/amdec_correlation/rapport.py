"""Rédaction du rapport Markdown : tableau récapitulatif, conclusions, seuils SPC.

Le tableau récapitulatif reprend exactement les lignes attendues au chapitre 10.
Une grandeur non calculable n'est jamais remplacée par une estimation : la
cellule porte la mention « non calculable » et le commentaire donne le motif
exact renvoyé par le calcul.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np

from . import metriques as M
from .analyse import ResultatCampagne, ResultatEssai

NON_CALCULABLE = "**non calculable**"


def _f(valeur: float, decimales: int = 3, signe: bool = False) -> str:
    if valeur is None or not math.isfinite(valeur):
        return NON_CALCULABLE
    format_ = f"{{:+.{decimales}f}}" if signe else f"{{:.{decimales}f}}"
    return format_.format(valeur)


def _motif(*candidats: str | None) -> str:
    for c in candidats:
        if c:
            return c
    return "grandeur non disponible"


def _majuscule(texte: str) -> str:
    """Met la première lettre en capitale, sans toucher au reste du texte.

    `str.capitalize()` mettrait tout le reste en minuscules : il abîmerait les
    sigles (PE, CRC, RMS) et les débuts de phrase des motifs qui en comportent
    plusieurs.
    """
    return texte[:1].upper() + texte[1:] if texte else texte


# ---------------------------------------------------------------------------
# Tableau récapitulatif
# ---------------------------------------------------------------------------


def tableau_recapitulatif(campagne: ResultatCampagne) -> str:
    cfg = campagne.config
    lignes: list[tuple[str, str, str]] = []

    # Le tableau lit les résultats CONSOLIDÉS : il n'a pas à savoir si les
    # essais ont été déclarés ou si leurs zones ont été découvertes.
    reg = campagne.regression_globale
    if reg is not None and reg.non_calculable:
        reg = None
    source = campagne.source_regression or "les paliers stabilisés"

    # 1. Offset b
    if reg:
        offset = 100.0 * reg.b / cfg.pleine_echelle_Nm
        incert = 100.0 * reg.sigma_b / cfg.pleine_echelle_Nm
        lignes.append((
            "Offset b (% PE)",
            _f(offset, 3, signe=True),
            f"Ordonnée à l'origine de la régression sur {source} : "
            f"{reg.b:+.2f} ± {reg.sigma_b:.2f} N·m (±{incert:.3f} % PE, k=1), "
            f"sur {reg.n} paliers stabilisés.",
        ))
    else:
        motif = _motif(
            campagne.regression_globale.non_calculable if campagne.regression_globale else None,
            "aucun palier stabilisé exploitable dans les acquisitions",
        )
        lignes.append(("Offset b (% PE)", NON_CALCULABLE, _majuscule(motif) + "."))

    # 2. Erreur de sensibilité
    if reg:
        lignes.append((
            "Erreur de sensibilité a−1 (%)",
            _f(100.0 * (reg.a - 1.0), 3, signe=True),
            f"Pente a = {reg.a:.5f} ± {reg.sigma_a:.5f} (k=1), R² = {reg.r2:.5f}. "
            f"Comparaison au point de mesure des transmissions, rapport de réduction "
            f"appliqué = {cfg.rapport_reduction:g}.",
        ))
    else:
        lignes.append(("Erreur de sensibilité a−1 (%)", NON_CALCULABLE,
                       "Découle de la même régression que l'offset, non disponible."))

    # 3. Non-linéarité
    if math.isfinite(campagne.non_linearite_pc_pe) and reg:
        lignes.append((
            "Non-linéarité (% PE)",
            _f(campagne.non_linearite_pc_pe, 3),
            f"Résidu maximal par rapport à la droite de régression "
            f"({reg.residu_max:.2f} N·m), écart-type des résidus "
            f"{reg.ecart_type_residus:.2f} N·m.",
        ))
    else:
        lignes.append(("Non-linéarité (% PE)", NON_CALCULABLE,
                       "Nécessite une régression exploitable sur des paliers stabilisés."))

    # 4. Hystérésis
    h = campagne.hysteresis_globale
    if h is not None and not h.non_calculable:
        lignes.append((
            "Hystérésis (% PE)",
            _f(h.max_pc_pe, 3),
            f"Écart maximal montée/descente au même couple de référence sur "
            f"{h.n_appariements} appariement(s) ; moyenne {h.moyenne_pc_pe:.3f} % PE. "
            f"Tolérance d'appariement {cfg.paliers.tolerance_appariement_pc_pe} % PE.",
        ))
    else:
        motif = _motif(
            h.non_calculable if h else None,
            "aucune acquisition ne présente montée et descente",
        )
        lignes.append(("Hystérésis (% PE)", NON_CALCULABLE, _majuscule(motif) + "."))

    # 5. Répétabilité après remontage
    rep_rem = campagne.repetabilite_remontage
    if rep_rem and not rep_rem.non_calculable:
        lignes.append((
            "Répétabilité après remontage (% PE)",
            _f(rep_rem.ecart_type_pc_pe, 3),
            f"Écart-type entre groupes de remontage à couple de référence constant, "
            f"{rep_rem.degres_liberte} ddl sur {len(rep_rem.groupes)} niveau(x).",
        ))
    else:
        motif = _motif(campagne.motif_remontage, rep_rem.non_calculable if rep_rem else None)
        rep_simple = (
            campagne.repetabilite_globale
            if campagne.repetabilite_globale and not campagne.repetabilite_globale.non_calculable
            else None
        )
        complement = (
            f" À titre indicatif, la répétabilité **sans remontage** (essais répétés à "
            f"température stabilisée) vaut {rep_simple.ecart_type_pc_pe:.3f} % PE "
            f"({rep_simple.degres_liberte} ddl) — ce n'est pas la même grandeur."
            if rep_simple else ""
        )
        lignes.append(("Répétabilité après remontage (% PE)", NON_CALCULABLE,
                       _majuscule(motif) + "." + complement))

    # 6. Retard temporel
    valides = [(c, r) for c, r in campagne.recalages_globaux if not r.non_calculable]
    retards = [r.retard_ms for _, r in valides]
    if retards:
        mediane = float(np.median(retards))
        apercu = valides[:6]
        detail = ", ".join(
            f"{Path(c).name} : {r.retard_ms:+.1f} ms"
            + (" ⚠" if r.faiblement_identifie else "")
            for c, r in apercu
        )
        if len(valides) > len(apercu):
            detail += f", … (+{len(valides) - len(apercu)})"
        # Une acquisition dont les sous-fenêtres ne s'accordent pas ne doit pas
        # se fondre dans une médiane sans que le rapport le dise : le chiffre
        # serait tenu pour mieux établi qu'il ne l'est.
        fragiles = [Path(c).name for c, r in valides if r.faiblement_identifie]
        reserve = (
            f" ⚠ {len(fragiles)} acquisition(s) au retard faiblement identifié "
            f"({', '.join(fragiles[:3])}"
            + (", …" if len(fragiles) > 3 else "")
            + ") : les sous-fenêtres de la plage corrélée y donnent des retards "
            f"s'étendant sur plus de "
            f"{cfg.intercorrelation.accord_blocs_max_ms:.0f} ms. Le retard n'y "
            "tient qu'à une partie du signal — typiquement le seul front d'un "
            "départ arrêté. À confronter aux essais à variations continues."
            if fragiles else ""
        )
        lignes.append((
            "Retard temporel (ms)",
            _f(mediane, 1, signe=True),
            f"Médiane sur {len(retards)} acquisition(s) dynamique(s) — {detail}. "
            f"Valeur positive = voie transmissions en retard sur la référence banc. "
            f"Recherche bornée à ±{cfg.intercorrelation.retard_max_ms:.0f} ms."
            + reserve,
        ))
    else:
        motif = _motif(
            *[r.non_calculable for _, r in campagne.recalages_globaux],
            "aucune acquisition ne présente de plage dynamique identifiable",
        )
        lignes.append(("Retard temporel (ms)", NON_CALCULABLE, _majuscule(motif) + "."))

    # 7. Sensibilité thermique
    th = campagne.thermique_globale
    if th and not th.non_calculable:
        lignes.append((
            "Sensibilité thermique (% PE pour 10 °C)",
            _f(th.pc_pe_pour_10C, 3, signe=True),
            f"Pente {th.pente_Nm_par_C:+.4f} N·m/°C ({th.pc_pe_par_C:+.5f} % PE/°C), "
            f"R² = {th.r2:.3f} sur {th.n_classes} cellules d'agrégation, excursion "
            f"{th.amplitude_C:.1f} °C. Régression sur moyennes par cellule pour éviter "
            f"l'autocorrélation des échantillons"
            + (
                f" ; régression multiple température + couple, l'effet du couple "
                f"(coefficient {th.coefficient_couple:+.4f} N·m/N·m) étant retiré pour que "
                f"l'erreur de gain ne soit pas comptée comme un effet thermique."
                if th.correction_couple else "."
            ),
        ))
    else:
        lignes.append(("Sensibilité thermique (% PE pour 10 °C)", NON_CALCULABLE,
                       _majuscule(_motif(th.non_calculable if th else None)) + "."))

    # 8. Dérive de zéro
    derives = campagne.derives_zero_globales
    if derives:
        pire_nom, pire = max(derives, key=lambda nd: abs(nd[1].derive_pc_pe))
        valeurs = [d.derive_pc_pe for _, d in derives]
        lignes.append((
            "Dérive de zéro sur cycle (% PE)",
            _f(pire.derive_pc_pe, 3, signe=True),
            f"Dérive la plus forte, sur « {pire_nom} » ({pire.zero_debut_Nm:+.2f} → "
            f"{pire.zero_fin_Nm:+.2f} N·m). Sur {len(derives)} essai(s) exploitable(s) : "
            f"moyenne {np.mean(valeurs):+.3f} % PE, étendue {np.ptp(valeurs):.3f} % PE.",
        ))
    else:
        motif = _motif("aucun relevé de zéro avant/après identifiable dans les acquisitions")
        lignes.append(("Dérive de zéro sur cycle (% PE)", NON_CALCULABLE, _majuscule(motif) + "."))

    # 9. Incertitude élargie
    inc = campagne.incertitude
    if inc and not inc.non_calculable:
        mention = (
            " ⚠️ **Minorant** : "
            + str(len(inc.exclusions))
            + " contribution(s) exclue(s) faute de donnée (voir § Bilan d'incertitude)."
            if inc.minorant else ""
        )
        lignes.append((
            "Incertitude élargie résultante (k=2)",
            f"{inc.U_k2_pc_pe:.3f} % PE ({inc.U_k2_Nm:.1f} N·m)",
            f"Somme quadratique de {len(inc.contributions)} contribution(s) supposées non "
            f"corrélées, u_c = {inc.u_composee_Nm:.2f} N·m, k = 2 (≈ 95 %).{mention}",
        ))
    else:
        lignes.append(("Incertitude élargie résultante (k=2)", NON_CALCULABLE,
                       _majuscule(_motif(inc.non_calculable if inc else None)) + "."))

    entete = "| Grandeur | Valeur mesurée | Commentaire |\n|---|---|---|\n"
    corps = "".join(f"| {g} | {v} | {c} |\n" for g, v, c in lignes)
    return entete + corps


# ---------------------------------------------------------------------------
# Sections du rapport
# ---------------------------------------------------------------------------


def section_conclusions(campagne: ResultatCampagne) -> str:
    dynamiques = campagne.par_type("dynamique")
    if not dynamiques:
        return "_Aucun essai dynamique déclaré en configuration._\n"
    blocs = []
    for e in dynamiques:
        if e.diagnostic is None:
            blocs.append(f"### {e.nom}\n\n_Essai non exploité : {_motif(*e.erreurs)}._\n")
            continue
        indices = " · ".join(
            f"{k} = {v:+.3f}" for k, v in e.diagnostic.indices.items() if math.isfinite(v)
        )
        blocs.append(
            f"### {e.nom}\n\n"
            f"> {e.diagnostic.conclusion}\n\n"
            f"_Cause retenue par la grille : **{e.diagnostic.cause}**. Indices : {indices}._\n"
        )
    return "\n".join(blocs)


def _mise_en_perspective_remontage(campagne: ResultatCampagne) -> str:
    """Situe l'écart de remontage par rapport à σ0 : c'est la cible de la carte."""
    spc, rep = campagne.spc, campagne.repetabilite_remontage
    if not (spc and not spc.non_calculable and spc.sigma0_pc_pe > 0):
        return ""
    if not (rep and not rep.non_calculable):
        return ""
    rapport_sigma = rep.ecart_type_pc_pe / spc.sigma0_pc_pe
    return (
        f"\n> **Dimensionnement vérifié.** La dispersion après remontage "
        f"({rep.ecart_type_pc_pe:.4f} % PE) vaut {rapport_sigma:.1f} σ0. Un remontage de la "
        f"chaîne de mesure produit donc un décalage que ces cartes détectent : les seuils "
        f"proposés ne sont ni trop serrés (ils absorbent la variabilité court terme) ni trop "
        f"larges (ils réagissent à un événement de remontage).\n"
        if rapport_sigma >= 1.0
        else (
            f"\n> **Dimensionnement vérifié.** La dispersion après remontage "
            f"({rep.ecart_type_pc_pe:.4f} % PE) reste sous σ0 ({rapport_sigma:.1f} σ0) : un "
            f"remontage ne produit pas de décalage détectable par ces cartes, ce qui est "
            f"favorable — le remontage n'est pas une source de dérive à surveiller.\n"
        )
    )


def section_spc(campagne: ResultatCampagne) -> str:
    spc = campagne.spc
    if spc is None or spc.non_calculable:
        return (
            f"{NON_CALCULABLE} — {_motif(spc.non_calculable if spc else None)}.\n\n"
            "Les paramètres μ0 et σ0 doivent être estimés sur des essais réellement répétés ; "
            "aucune valeur par défaut n'est proposée ici, elle serait arbitraire.\n"
        )
    return f"""Les paramètres de position et de dispersion sont **déduits des essais répétés**
(résidu mesuré − référence aux paliers stabilisés, exprimé en % PE), centrés par niveau
de couple afin que la dispersion estimée soit bien celle de la chaîne de mesure à point
de fonctionnement donné :

| Paramètre | Valeur | Origine |
|---|---|---|
| μ0 (centre) | {spc.mu0_pc_pe:+.4f} % PE | moyenne du résidu sur {spc.n} paliers d'essais répétés |
| σ0 (dispersion) | {spc.sigma0_pc_pe:.4f} % PE | répétabilité poolée **à couple de référence constant**, {spc.ddl} ddl |
| σ0 robuste (étendue mobile) | {spc.sigma0_robuste_pc_pe:.4f} % PE | MR moyen / d₂ (d₂ = {M.D2_ETENDUE_MOBILE}) — insensible à une dérive lente ; sert de contre-épreuve à σ0 |
{_mise_en_perspective_remontage(campagne)}

**Carte CUSUM** (détection d'un décalage de 1 σ) :

| Paramètre | Valeur | Justification |
|---|---|---|
| k (référence) | {spc.cusum_k_pc_pe:.4f} % PE | k = 0,5·σ0 — optimal pour un décalage de 1 σ |
| h **alerte** | {spc.cusum_h_alerte_pc_pe:.4f} % PE | h = 4·σ0 → ARL0 ≈ 168 |
| h **alarme** | {spc.cusum_h_alarme_pc_pe:.4f} % PE | h = 5·σ0 → ARL0 ≈ 465 |

**Carte EWMA** :

| Paramètre | Valeur | Justification |
|---|---|---|
| λ | {spc.ewma_lambda:.2f} | compromis usuel pour détecter des décalages de 0,5 à 1 σ |
| σ_EWMA(∞) | {spc.ewma_sigma_asymptotique_pc_pe:.4f} % PE | σ0·√(λ/(2−λ)) |
| L **alerte** | {spc.ewma_L_alerte:.2f} → ±{spc.ewma_limite_alerte_pc_pe:.4f} % PE | limite à ≈ 2 σ |
| L **alarme** | {spc.ewma_L_alarme:.3f} → ±{spc.ewma_limite_alarme_pc_pe:.4f} % PE | ARL0 ≈ 370 pour λ = {spc.ewma_lambda:.2f} |

Limites de contrôle EWMA à appliquer : **{spc.mu0_pc_pe:+.4f} ± {spc.ewma_limite_alarme_pc_pe:.4f} % PE**
(alarme) et **{spc.mu0_pc_pe:+.4f} ± {spc.ewma_limite_alerte_pc_pe:.4f} % PE** (alerte).

> μ0 et σ0 sortent des mesures. Les multiplicateurs (k = 0,5σ ; h = 4σ / 5σ ; λ = 0,20 ;
> L = 2,962) sont les valeurs tabulées correspondant aux ARL0 usuelles en maîtrise
> statistique des procédés (Montgomery, *Introduction to Statistical Quality Control*) :
> ce ne sont pas des choix arbitraires, et ils restent valables tant que σ0 est
> ré-estimé sur des essais répétés.

> ⚠️ **Condition d'emploi de μ0.** Dès lors qu'une erreur de sensibilité subsiste, le
> résidu dépend du couple appliqué : μ0 n'a de sens que si la surveillance porte sur des
> points de fonctionnement comparables à ceux des essais répétés ayant servi à l'estimer.
> Pour une surveillance sur plage de couple étendue, cartographier le résidu à couple de
> référence fixé, ou porter en carte le résidu **après correction de gain**, dont la
> valeur cible est alors nulle. σ0 n'est pas concerné : il est estimé à couple constant.
"""


def section_incertitude(campagne: ResultatCampagne) -> str:
    inc = campagne.incertitude
    if inc is None or inc.non_calculable:
        return f"{NON_CALCULABLE} — {_motif(inc.non_calculable if inc else None)}.\n"
    cfg = campagne.config
    lignes = "".join(
        f"| {c.nom} | {c.valeur_Nm:.3f} | {c.loi} | {c.u_Nm:.3f} | "
        f"{100 * c.u_Nm / cfg.pleine_echelle_Nm:.4f} | {c.commentaire} |\n"
        for c in inc.contributions
    )
    texte = f"""| Contribution | Valeur (N·m) | Loi | u (N·m) | u (% PE) | Origine |
|---|---|---|---|---|---|
{lignes}| **u_c (composée)** | | | **{inc.u_composee_Nm:.3f}** | **{100 * inc.u_composee_Nm / cfg.pleine_echelle_Nm:.4f}** | somme quadratique |
| **U (k = 2)** | | | **{inc.U_k2_Nm:.3f}** | **{inc.U_k2_pc_pe:.4f}** | ≈ 95 % de confiance |
"""
    if inc.exclusions:
        texte += (
            "\n**Contributions exclues du bilan** — l'incertitude ci-dessus est donc un "
            "**minorant** :\n\n"
            + "".join(f"- {e}\n" for e in inc.exclusions)
        )
    return texte


def section_detail_acquisitions(campagne: ResultatCampagne) -> str:
    """Relevé acquisition par acquisition, en organisation automatique.

    Aucun essai n'étant déclaré, il n'y a pas de « détail par essai » à rédiger :
    ce qui doit être vérifiable, c'est **ce qui a été trouvé dans chaque fichier**
    et ce que ce fichier a effectivement alimenté. Sans ce relevé, l'organisation
    automatique serait opaque — et un chiffre qu'on ne peut pas remonter à sa
    source n'a pas sa place dans un rapport relu par des tiers.
    """
    if not campagne.zones:
        return "_Aucune acquisition analysée._\n"

    recalages = {chemin.name: rec for chemin, rec in campagne.recalages_globaux}
    derives = dict(campagne.derives_zero_globales)

    lignes = [
        "| Acquisition | Durée (s) | Paliers | Niveaux | ↑ et ↓ | Dynamique (s) "
        "| Zéros déb./fin | Alimente |",
        "|---|---:|---:|---:|:---:|---:|:---:|---|",
    ]
    for z in campagne.zones:
        if z.erreur:
            lignes.append(
                f"| `{z.chemin.name}` | — | — | — | — | — | — | "
                f"**non exploitée** — {z.erreur} |"
            )
            continue
        dynamique = f"{z.duree_dynamique_s:.0f}" if z.alimente_retard else "—"
        lignes.append(
            f"| `{z.chemin.name}` | {z.duree_s:.0f} | {len(z.paliers)} "
            f"| {z.niveaux_distincts} | {'oui' if z.a_montee_et_descente else '—'} "
            f"| {dynamique} | {'oui' if z.alimente_derive_zero else '—'} "
            f"| {', '.join(z.contributions()) or '—'} |"
        )

    detail = []
    for nom, rec in sorted(recalages.items()):
        if rec.non_calculable:
            detail.append(f"- Recalage `{nom}` : {rec.non_calculable}")
        else:
            reserve = (
                f" — ⚠ **faiblement identifié** : les sous-fenêtres donnent "
                f"{rec.dispersion_blocs_ms:.0f} ms d'étendue"
                if rec.faiblement_identifie else ""
            )
            fenetres = (
                f"{len(rec.fenetres)} fenêtres, {rec.duree_fenetre_s:.0f} s au total"
                if len(rec.fenetres) > 1
                else f"1 fenêtre de {rec.duree_fenetre_s:.0f} s "
                     f"[{rec.fenetre[0]:.0f}–{rec.fenetre[1]:.0f} s]"
            )
            detail.append(
                f"- Recalage `{nom}` : {rec.retard_ms:+.1f} ms "
                f"(médiane sur {fenetres}, r = {rec.correlation_pic:.3f}, "
                f"dispersion {rec.dispersion_blocs_ms:.1f} ms {rec.source_dispersion}, "
                f"passe-haut {rec.coupure_passe_haut_Hz:.2f} Hz, RMS résidu "
                f"{rec.rms_residu_avant_Nm:.1f} → {rec.rms_residu_apres_Nm:.1f} N·m)"
                + reserve
            )
            if len(rec.fenetres) > 1:
                detail.append(
                    "  - par fenêtre : "
                    + " ; ".join(
                        f"{f[0]:.0f}–{f[1]:.0f} s : {retard:+.1f} ms"
                        for f, retard, _ in rec.fenetres
                    )
                )
    for nom, d in sorted(derives.items()):
        detail.append(
            f"- Dérive de zéro `{nom}` : {d.derive_pc_pe:+.3f} % PE "
            f"({d.zero_debut_Nm:+.2f} → {d.zero_fin_Nm:+.2f} N·m)"
            if not d.non_calculable
            else f"- Dérive de zéro `{nom}` : {d.non_calculable}"
        )

    texte = "\n".join(lignes) + "\n"
    if detail:
        texte += "\n" + "\n".join(detail) + "\n"
    return texte


def section_detail_essais(campagne: ResultatCampagne) -> str:
    if campagne.mode == "automatique":
        return section_detail_acquisitions(campagne)

    cfg = campagne.config
    blocs = []
    for e in campagne.essais:
        titre = f"### {e.nom}  \n_type : `{e.declaration.type}`_\n"
        if not e.fichiers_traites:
            blocs.append(titre + f"\n**Essai non exploité.** {_motif(*e.erreurs)}\n")
            continue

        lignes = [f"- Fichiers traités : {len(e.fichiers_traites)}"]
        for avertissement in e.avertissements:
            lignes.append(f"- ⚠️ {avertissement}")
        if e.erreurs:
            lignes.append(f"- Fichiers en erreur : {len(e.erreurs)} — " + " ; ".join(e.erreurs[:3]))
        if e.paliers:
            montees = sum(1 for p in e.paliers if p.sens == "montee")
            lignes.append(
                f"- Paliers stabilisés détectés : {len(e.paliers)} "
                f"({montees} en montée, {len(e.paliers) - montees} en descente)"
            )
        if e.regression and not e.regression.non_calculable:
            r = e.regression
            lignes.append(
                f"- Régression : a = {r.a:.5f} ± {r.sigma_a:.5f}, b = {r.b:+.2f} ± {r.sigma_b:.2f} N·m, "
                f"R² = {r.r2:.5f}"
            )
        elif e.regression:
            lignes.append(f"- Régression : {e.regression.non_calculable}")
        if e.repetabilite:
            lignes.append(
                f"- Répétabilité : σ = {e.repetabilite.ecart_type_pc_pe:.4f} % PE "
                f"({e.repetabilite.degres_liberte} ddl)"
                if not e.repetabilite.non_calculable
                else f"- Répétabilité : {e.repetabilite.non_calculable}"
            )
        for chemin, rec in e.recalages:
            if rec.non_calculable:
                lignes.append(f"- Recalage `{chemin.name}` : {rec.non_calculable}")
            else:
                lignes.append(
                    f"- Recalage `{chemin.name}` : {rec.retard_ms:+.1f} ms "
                    f"(r = {rec.correlation_pic:.3f}, fenêtre {rec.duree_fenetre_s:.0f} s, "
                    f"RMS résidu {rec.rms_residu_avant_Nm:.1f} → {rec.rms_residu_apres_Nm:.1f} N·m)"
                )
        if e.thermique:
            lignes.append(
                f"- Sensibilité thermique : {e.thermique.pc_pe_pour_10C:+.3f} % PE / 10 °C "
                f"(R² = {e.thermique.r2:.3f})"
                if not e.thermique.non_calculable
                else f"- Sensibilité thermique : {e.thermique.non_calculable}"
            )
        for chemin, d in e.derives_zero:
            lignes.append(
                f"- Dérive de zéro `{chemin.name}` : {d.derive_pc_pe:+.3f} % PE "
                f"({d.zero_debut_Nm:+.2f} → {d.zero_fin_Nm:+.2f} N·m)"
                if not d.non_calculable
                else f"- Dérive de zéro `{chemin.name}` : {d.non_calculable}"
            )
        if e.redondance:
            lignes.append(
                f"- Redondance gauche−droite : moyenne {e.redondance.moyenne_pc_pe:+.3f} % PE, "
                f"σ = {e.redondance.ecart_type_pc_pe:.3f} % PE, "
                f"max {e.redondance.max_absolu_pc_pe:.3f} % PE"
                if not e.redondance.non_calculable
                else f"- Redondance gauche−droite : {e.redondance.non_calculable}"
            )
        if e.diagnostics_temps:
            lignes.append(f"- Base de temps : {e.diagnostics_temps[0].split('— ', 1)[-1]}")
        for fichier, etats in e.etats.items():
            for canal, distribution in etats.items():
                repartition = ", ".join(f"`{v}` : {p:.2f} %" for v, p in list(distribution.items())[:6])
                lignes.append(
                    f"- Canal d'état `{canal}` ({fichier}) : {repartition} "
                    f"— _la valeur nominale n'est pas connue du pipeline, à interpréter par l'exploitant_"
                )
        if e.figures:
            lignes.append("- Figures : " + ", ".join(f"`{f.name}`" for f in e.figures))
        blocs.append(titre + "\n" + "\n".join(lignes) + "\n")
    return "\n".join(blocs)


def section_hypotheses(campagne: ResultatCampagne) -> str:
    cfg = campagne.config
    return f"""Toutes les valeurs ci-dessous sont issues de la configuration et non de choix
implicites du code.

| Hypothèse | Valeur retenue |
|---|---|
| Pleine échelle capteur (PE) | {cfg.pleine_echelle_Nm:g} N·m |
| Voie de couple confrontée à la référence | `{cfg.mode_comparaison}` |
| Rapport de réduction appliqué à la référence | {cfg.rapport_reduction:g} |
| Fréquence de rééchantillonnage sur base de temps commune | {cfg.frequence_Hz:g} Hz |
| Durée de fenêtre de stabilisation (paliers) | {cfg.paliers.duree_palier_s:g} s |
| Critère de stabilisation (écart-type du couple de référence) | {cfg.paliers.tolerance_stab_pc_pe:g} % PE |
| Fraction finale du palier moyennée | {cfg.paliers.fraction_finale:g} |
| Écart minimal entre deux paliers distincts | {cfg.paliers.ecart_min_paliers_pc_pe:g} % PE |
| Tolérance d'appariement montée/descente et de regroupement | {cfg.paliers.tolerance_appariement_pc_pe:g} % PE |
| Bornes de recherche du retard (intercorrélation) | ±{cfg.intercorrelation.retard_max_ms:g} ms |
| Filtre passe-haut avant intercorrélation | {cfg.intercorrelation.passe_haut_Hz} Hz (Butterworth ordre 2, phase nulle) |
| Seuil d'activité dynamique pour l'intercorrélation | {cfg.intercorrelation.seuil_activite_pc_pe:g} % PE |
| Fenêtre de relevé de zéro | {cfg.zero.duree_fenetre_s:g} s, |C_réf| < {cfg.zero.seuil_couple_ref_pc_pe:g} % PE, |N| < {cfg.zero.seuil_regime} |
| Position exigée des relevés de zéro | premiers / derniers {cfg.zero.fraction_bord * 100:.0f} % de l'essai |
| Excursion thermique minimale pour identifier une pente | {cfg.thermique.amplitude_min_C:g} °C |
| Plage de température de service (bilan d'incertitude) | {cfg.thermique.plage_service_C if cfg.thermique.plage_service_C else "non renseignée → contribution exclue"} |
| Incertitude-type de la référence banc (k=1) | {f"{cfg.incertitude_reference_k1_Nm:g} N·m" if cfg.incertitude_reference_k1_Nm is not None else "non renseignée → contribution exclue"} |
| Facteur d'élargissement | k = 2 (≈ 95 %) |
"""


def rediger(campagne: ResultatCampagne) -> str:
    cfg = campagne.config
    horodatage = datetime.now().strftime("%Y-%m-%d %H:%M")
    figures = campagne.figures + [f for e in campagne.essais for f in e.figures]

    avertissements = (
        "\n".join(f"- {a}" for a in campagne.avertissements)
        if campagne.avertissements
        else (
            "_Aucun : tous les canaux attendus sont mappés sur toutes les acquisitions, "
            "et toutes les saisies utilisateur sont renseignées._"
        )
    )
    liste_figures = (
        "\n".join(f"- `figures/{f.name}`" for f in figures)
        if figures
        else "_Aucune figure produite._"
    )
    titre_detail = (
        "Détail par acquisition" if campagne.mode == "automatique" else "Détail par essai"
    )

    return f"""# Chapitre 10 — Apport de la campagne de corrélation sur banc GMP

_Rapport généré le {horodatage} — racine des données : `{cfg.racine}`_

Comparaison entre le couple mesuré par les transmissions instrumentées
(télémétrie Manner PCM16) et le couple de référence mesuré sur banc GMP.
Le **résidu** est défini partout comme `couple_mesuré − couple_référence`.
« % PE » rapporte la grandeur à la pleine échelle de {cfg.pleine_echelle_Nm:g} N·m.

## 10.1 Tableau récapitulatif

{tableau_recapitulatif(campagne)}
## 10.2 Conclusions par essai dynamique

{section_conclusions(campagne)}
## 10.3 Bilan d'incertitude

{section_incertitude(campagne)}
## 10.4 Paramètres proposés pour les cartes de contrôle (chapitre 11)

{section_spc(campagne)}
## 10.5 {titre_detail}

{section_detail_essais(campagne)}
## 10.6 Hypothèses de traitement

{section_hypotheses(campagne)}
## 10.7 Avertissements (mapping et saisies utilisateur)

{avertissements}

## 10.8 Figures produites

{liste_figures}

---

_Les acquisitions `.mf4` ont été ouvertes en lecture seule (handle `rb`) ; aucune source
n'a été modifiée. Les grandeurs marquées « non calculable » ne sont pas estimées :
le motif exact est donné en commentaire._
"""


def ecrire(campagne: ResultatCampagne, chemin: Path | None = None) -> Path:
    chemin = chemin or (campagne.config.dossier_sortie / "chapitre10_correlation.md")
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(rediger(campagne), encoding="utf-8")
    return chemin
