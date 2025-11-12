import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import layers
from keras import Sequential



# Architecture du réseau
modele = Sequential()
# Couches de neurones
modele.add(layers.Input(shape=(1,)))
modele.add(layers.Dense(2, activation="relu"))
modele.add(layers.Dense(1, activation="relu"))

# Pour vérifier l'achitecture du réseau
modele.summary()




import numpy as np

# Couche 0 (1->2)
coeff = np.array([[1., -0.5]]) # (1, 2)
biais = np.array([-1, 1]) # (2,)
modele.layers[0].set_weights([coeff, biais])



# Poids de la deuxième couche (2->1)
coeff2 = np.array([[1], [1]]) # (2, 1)
biais2 = np.array([0]) # (1,)
modele.layers[1].set_weights([coeff2, biais2])

modele.layers[0].get_weights()









entree = np.array([[3.0]])
sortie = modele.predict(entree)
print(sortie)










import matplotlib.pyplot as plt

liste_x = np.linspace(-2, 3, num=100)
entree = np.array([[x] for x in liste_x])
sortie = modele.predict(entree)
liste_y = np.array([y[0] for y in sortie])
plt.plot(liste_x,liste_y)
plt.show()

