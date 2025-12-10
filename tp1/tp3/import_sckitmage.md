# Installation de skimage dans le home
```
mkdir -p ~/py-packages
pip install --target ~/py-packages scikit-image
```

# Ouvrir l'environnement /opt/pytorch-env/bin/activate
```
source /opt/pytorch-env/bin/activate
```
# Dans le notebook, il faut ajouter le chemin vers le dossier py-packages
```
import os
import sys
sys.path.append(os.path.expanduser("~/py-packages/"))
```