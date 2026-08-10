"""Interface graphique locale du pipeline de corrélation couple.

L'interface est une **coquille au-dessus de la bibliothèque** : elle ne
contient aucun traitement. Elle assemble une configuration, appelle
`analyse.analyser` puis `rapport.rediger`, et affiche ce qui en sort. Les
chiffres affichés à l'écran sont donc, par construction, ceux du rapport.

Lancement :

    python scripts/03_interface.py

Tout s'exécute en local : les acquisitions ne quittent pas le poste.
"""

from . import etat

__all__ = ["etat"]
