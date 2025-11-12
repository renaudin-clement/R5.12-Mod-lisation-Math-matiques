import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import layers
from keras import Sequential



# Architecture du réseau
modele = Sequential()
# Couches de neurones
modele.add(layers.Input(shape=(2,)))
modele.add(layers.Dense(3, activation="sigmoid"))
modele.add(layers.Dense(1, activation="sigmoid"))

# Pour vérifier l'achitecture du réseau
modele.summary()




import numpy as np

# Couche 0 (1->2)
coeff = np.array([[1, 3,-5],[2,-4,-6]]) # (1, 2)
biais = np.array([-1, 0,1]) # (2,)
modele.layers[0].set_weights([coeff, biais])



# Poids de la deuxième couche (2->1)
coeff2 = np.array([[1], [1],[1]]) # (2, 1)
biais2 = np.array([-3]) # (1,)
modele.layers[1].set_weights([coeff2, biais2])



entree = np.array([[7,-5]])
sortie = modele.predict(entree)
print(sortie)







# --- Visualisation 3D du modèle ---
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Création d'une grille de valeurs pour les axes X et Y
VX = np.linspace(-5, 5, 20)
VY = np.linspace(-5, 5, 20)
# Construction du maillage 2D (toutes les combinaisons X, Y)
X, Y = np.meshgrid(VX, VY)
# Préparation des données d'entrée pour le modèle (couples x, y)
entree = np.c_[X.ravel(), Y.ravel()]
# Prédiction du modèle sur chaque point de la grille
Z = modele.predict(entree).reshape(X.shape)
# Création de la figure 3D
fig = plt.figure()
ax = plt.axes(projection='3d')
# Tracé de la surface prédite
ax.plot_surface(X, Y, Z, cmap='viridis')
# Affichage du graphique
plt.show()



