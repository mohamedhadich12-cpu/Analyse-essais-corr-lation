"""Corrélation couple transmissions instrumentées (Manner PCM16) / banc GMP.

Pipeline d'analyse des acquisitions MDF4 pour le chapitre 10 du rapport AMDEC.

Principes tenus par l'ensemble du paquet :

* **Lecture seule stricte** : les fichiers `.mf4` sources ne sont jamais
  modifiés ni réécrits (cf. `io_mdf.ouvrir_mdf`).
* **Aucune valeur inventée** : toute grandeur qui ne peut pas être calculée
  faute de canal, de plage ou de répétition disponible est renvoyée avec un
  motif explicite de non-calculabilité, et apparaît comme telle dans le
  rapport. Aucune extrapolation n'est faite.
* **Hypothèses explicites** : toutes les bornes de traitement (durée des
  paliers stabilisés, fenêtre d'intercorrélation, critères de détection du
  zéro, ...) sont dans la configuration, jamais codées en dur.
"""

__all__ = [
    "config",
    "io_mdf",
    "inventaire",
    "metriques",
    "graphiques",
    "rapport",
    "analyse",
]

__version__ = "1.0.0"
