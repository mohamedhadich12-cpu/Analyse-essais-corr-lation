# Corrélation couple transmissions instrumentées / banc GMP

Pipeline d'analyse des acquisitions MDF4 pour le **chapitre 10** du rapport AMDEC
(« Apport de la campagne de corrélation sur banc GMP »), et pour le dimensionnement
des cartes de contrôle du **chapitre 11**.

Formats d'acquisition lus : **`.mf4`, `.mdf`** et **`.aif`** (exports ETAS INCA,
qui sont des conteneurs MDF sous une extension propre à l'outil). L'appartenance
au format est vérifiée sur le **contenu** du fichier, jamais sur son nom.

Il compare le couple mesuré par les transmissions instrumentées (télémétrie
Manner PCM16) au couple de référence mesuré sur banc GMP, et produit :

* un **tableau récapitulatif Markdown** aux lignes exactes attendues au chapitre 10 ;
* une **conclusion qualitative par essai dynamique**, directement collable dans le
  rapport, formulée selon la grille offset / gain / synchronisation / écart de modèle ;
* les **paramètres CUSUM et EWMA** (μ0, σ0, k, h, λ, L) déduits de la dispersion
  réellement mesurée ;
* les **figures PNG** prêtes à insérer.

## Le pipeline tourne sur votre poste

Les acquisitions ne quittent pas votre machine : tout s'exécute en local, en lecture
seule sur les `.mf4`. Rien dans ce dépôt ne contient de données d'essai
(`.mf4`, `.mdf` et `.dat` sont exclus par `.gitignore`).

## Guide utilisateur

**[`docs/Guide_correlation_couple_banc_GMP.pdf`](docs/Guide_correlation_couple_banc_GMP.pdf)** —
23 pages : l'interface onglet par onglet, les types d'essai, chaque indicateur
(définition, méthode, hypothèse structurante, comment le lire), le bilan
d'incertitude, les cartes de contrôle et un glossaire. À lire avant la première
utilisation.

Pour le régénérer après une évolution de l'interface : `python docs/source/captures.py`
puis `python docs/construire_guide.py` (voir l'entête de chaque script).

## Installation

**Python 3.10 à 3.13.** `asammdf` dépend de `zstd`, dont les wheels Windows
s'arrêtent à cp313 : sur Python 3.14+, pip tente de compiler le module C et
échoue sur `Microsoft Visual C++ 14.0 or greater is required`. Sous Windows :

```powershell
py -0                      # liste les versions installées
py -3.13 -m venv .venv     # créer l'environnement avec une version prise en charge
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Rétrograder `asammdf` ne contourne pas le problème : ses versions antérieures
épinglent `numpy<2.0`, qui n'a pas non plus de wheels au-delà de cp312.

> Si l'import d'`asammdf` échoue avec `No module named '_cffi_backend'` :
> `python -m pip install --force-reinstall cffi`

Vérification de l'installation, sans toucher aux données réelles :

```bash
python tests/generer_mf4_synthetique.py --sortie demo/donnees
python -m pytest tests/ -q
```

Le générateur fabrique une arborescence d'essais **synthétiques à vérité connue**
(gain, offset, hystérésis, retard, dérive thermique, biais de remontage injectés) ;
les tests vérifient que le pipeline retrouve bien chacune de ces valeurs. Aucun
chiffre issu de ces fichiers ne doit figurer dans le rapport.

## Utilisation — interface graphique (recommandé)

```bash
python scripts/03_interface.py
```

Une page s'ouvre dans le navigateur, sur `localhost` uniquement. Cinq onglets
suivent le déroulé de l'analyse :

| Onglet | Ce qu'on y fait |
|---|---|
| **1 · Exploration** | lit les acquisitions et liste les canaux présents, avec unités, cadences et diagnostic de base de temps |
| **2 · Canaux** | associe chaque rôle à un canal réel, par **liste déroulante peuplée des noms trouvés** — plus de libellé à recopier. Par défaut le mapping est **commun à toute la campagne** : les libellés étant généralement identiques d'un essai à l'autre, il n'y a aucune raison de les redéclarer dossier par dossier. Décocher la case rétablit un mapping par dossier |
| **3 · Essais** | type d'essai, ligne droite, groupe de remontage |
| **4 · Hypothèses** | toutes les bornes de traitement, et les deux saisies utilisateur |
| **5 · Analyse & résultats** | lance l'analyse, puis affiche tableau récapitulatif, conclusions, figures, bilan d'incertitude, paramètres de cartes de contrôle et détail par essai |

L'interface **ne contient aucun traitement** : elle assemble une configuration,
appelle la même bibliothèque que la ligne de commande et affiche ce qui en sort.
La configuration se télécharge en `.yaml` et rejoue à l'identique :

```bash
python scripts/02_analyse.py --config correlation.yaml
```

Un test le vérifie explicitement — écran et ligne de commande ne sont pas deux
chemins de calcul.

## Utilisation — ligne de commande

### Étape 1 — inventorier les canaux

```bash
python scripts/01_inventaire.py --racine "C:/Users/SD17365/Documents"
```

Écrit dans `sortie/inventaire/` :

| Fichier | Contenu |
|---|---|
| `inventaire_canaux.md` | tous les canaux par sous-dossier : nom, unité, groupe, cadence, plage de temps, et un diagnostic de base de temps |
| _(rappel)_ | l'inventaire couvre `.mf4`, `.mdf` et `.aif`, casse de l'extension indifférente |
| `canaux.csv` | le même à plat, pour recherche |
| `canaux_proposition.yaml` | squelette de mapping pré-rempli avec des **candidats à valider** |

Les candidats sont repérés par simple correspondance de mots-clés et marqués
`# À VÉRIFIER`. **Aucun n'est utilisé tant qu'il n'a pas été recopié à la main**
dans la configuration : une erreur de mapping fausserait tous les résultats.

### Étape 2 — renseigner la configuration

```bash
cp config/correlation.example.yaml config/correlation.yaml
```

Puis renseigner, en s'appuyant sur le rapport d'inventaire, tout ce qui est marqué
`À RENSEIGNER` ou `À CONFIRMER` — notamment :

| Clé | Pourquoi elle ne peut pas être devinée |
|---|---|
| `canaux:` | les libellés réels diffèrent d'une campagne à l'autre. La référence banc se déclare sur **deux voies** (`couple_reference_gauche` / `_droite`) ; `couple_reference` n'est qu'un repli si le banc n'en fournit qu'une |
| `comparaison.mode` | s'applique **de la même façon** aux voies mesurées et aux voies de référence : en `moyenne`, (G+D)/2 mesuré est confronté à (G+D)/2 de référence |
| `essais.<dossier>.type` | `balayage`, `repetabilite` ou `dynamique` — détermine le traitement |
| `essais.<dossier>.ligne_droite` | conditionne l'exploitation du résidu gauche − droite : hors ligne droite, cet écart est physique et non métrologique |
| `remontage.realise` / `essais.<dossier>.groupe_remontage` | il faut **deux** étiquettes distinctes pour que la répétabilité après remontage existe. `remontage.realise: false` déclare qu'aucune dépose/repose n'a eu lieu : le rapport motive alors la non-calculabilité par « grandeur non définie » et non par « clé non configurée » |
| `comparaison.incertitude_reference_k1_Nm` | vient du certificat d'étalonnage du banc, pas des acquisitions |
| `thermique.plage_service_C` | plage de température en service, pour convertir la sensibilité thermique en contribution d'incertitude |

Sans ces deux dernières, le bilan d'incertitude reste calculé mais est explicitement
annoncé comme un **minorant**, avec la liste des contributions exclues.

### Étape 3 — lancer l'analyse

```bash
python scripts/02_analyse.py --config config/correlation.yaml
```

Écrit `sortie/chapitre10_correlation.md` et `sortie/figures/*.png`.

## Ce qui est calculé, et sous quelles hypothèses

| Grandeur | Méthode | Hypothèse structurante |
|---|---|---|
| Offset `b`, sensibilité `a−1` | régression linéaire sur les paliers stabilisés | palier = fenêtre où l'écart-type du **couple de référence** reste sous le seuil ; seule la fraction finale est moyennée |
| Non-linéarité | résidu maximal à la droite de régression | — |
| Hystérésis | écart montée/descente au même couple de référence | appariement à tolérance donnée ; sens déduit de la variation de niveau |
| Répétabilité | écart-type poolé **du résidu** à couple constant | raisonner sur le résidu retire la variation résiduelle de la référence |
| Retard temporel | intercorrélation, pic affiné au sous-échantillon | passe-haut à phase nulle préalable ; recherche bornée ; seule la plage dynamique la plus longue est corrélée |
| Sensibilité thermique | régression **multiple** `résidu = α·couple + β·T + γ` | β seul est l'effet thermique : sans cette séparation, l'erreur de gain serait comptée comme thermique, couple et température montant ensemble au fil d'un essai |
| Dérive de zéro | écart entre relevés de repos de début et de fin | le repos doit tomber dans les premiers / derniers % de l'essai |
| Redondance G−D | moyenne et écart-type de l'écart entre voies | valable seulement en ligne droite sans sollicitation différentielle |
| Incertitude élargie | somme quadratique, `k = 2` | contributions supposées non corrélées ; lois rectangulaires pour les grandeurs bornées |
| μ0, σ0, seuils SPC | dispersion des essais répétés | σ0 = variabilité **court terme** (à couple constant, sans démontage) : les décalages entre essais sont ce que la carte doit détecter, pas ce qui doit élargir ses limites |

Toutes les bornes numériques de ces hypothèses sont dans la configuration, jamais
codées en dur, et sont reprises telles quelles dans le § « Hypothèses de traitement »
du rapport généré.

Les couples (λ, L) et multiplicateurs CUSUM proviennent des tables ARL usuelles
(Montgomery, *Introduction to Statistical Quality Control*) ; seuls μ0 et σ0 sortent
des mesures. Aucun seuil n'est posé arbitrairement.

## Deux garanties tenues par le code

**Lecture seule stricte.** Chaque `.mf4` est ouvert via un handle Python `'rb'` :
l'écriture est impossible au niveau du système de fichiers, et aucune méthode mutante
d'asammdf n'est appelée. Un test compare les empreintes SHA-256 et les dates de
modification de tous les fichiers avant et après une analyse complète.

**Un même essai n'est jamais compté deux fois sans le dire.** Si une acquisition existe
sous plusieurs formats dans le même dossier (`releve_01.mf4` et `releve_01.aif`), les deux
fichiers portent les mêmes grandeurs et seraient traités comme des essais distincts —
faussant tout ce qui se cumule, sans qu'aucun calcul n'échoue. Le doublon est détecté et
signalé, à l'écran comme dans le rapport.

**Aucune valeur inventée.** Toute grandeur qui ne peut pas être obtenue — canal absent,
excursion thermique insuffisante, pas de relevé de zéro identifiable, aucun essai
répété — est rendue avec un **motif explicite de non-calculabilité**, repris dans le
rapport. Rien n'est extrapolé, et aucune valeur par défaut ne se substitue à une mesure
manquante.

## Structure

```
config/correlation.example.yaml   configuration commentée, à copier
.streamlit/config.toml            thème de l'interface (mêmes couleurs que les figures)
scripts/01_inventaire.py          étape 1 : inventaire des canaux
scripts/02_analyse.py             étapes 2-3 : traitements, rapport, figures
scripts/03_interface.py           lance l'interface graphique locale
src/amdec_correlation/
  config.py       chargement et validation ; toutes les hypothèses de traitement
  io_mdf.py       lecture seule MDF4, rééchantillonnage sur base de temps commune
  inventaire.py   étape 1
  metriques.py    régression, paliers, hystérésis, recalage, thermique, zéro, incertitude, SPC
  graphiques.py   figures PNG (charte : une couleur = une entité, marques fines)
  rapport.py      rédaction du Markdown
  analyse.py      orchestration par type d'essai
  interface/
    app.py        page Streamlit (affichage uniquement)
    etat.py       assemblage de la configuration, sans dépendance à Streamlit
tests/
  generer_mf4_synthetique.py  arborescence synthétique à vérité connue
  test_metriques.py           validation des formules
  test_bout_en_bout.py        validation du pipeline complet
  test_interface.py           logique de l'interface, sans lancer Streamlit
```

## Bases de temps

Dans un MDF4, chaque groupe de canaux possède son propre canal maître : les canaux ne
partagent pas nécessairement la même base de temps. Le pipeline ramène tous les canaux
mappés sur une grille commune uniforme (intersection des plages de temps, aucune
extrapolation), interpolation linéaire pour les grandeurs physiques et maintien de la
dernière valeur pour les canaux d'état. Le constat est restitué essai par essai dans le
rapport, et l'inventaire signale d'emblée les fichiers à plusieurs groupes.

## Canaux d'état (LED, CRC, erreurs)

Ils sont relevés et leur distribution de valeurs est restituée, **sans interprétation** :
la valeur nominale d'une LED télémétrie ou d'un compteur CRC n'est pas connue du
pipeline. C'est à l'exploitant de dire quelle valeur est nominale.
