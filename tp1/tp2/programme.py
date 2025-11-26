import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap
import math as maths

# Création d’une image 2x2
image = np.array([[0.9, 0.2],[0.1 , 0.8]])

imageV =   [
            0.9,
            0.2,
            0.1,
            0.8,
            ]

# Affichage
cmp = get_cmap('gray')
plt.imshow(image, cmap=cmp, vmin=0, vmax=1)
plt.show()
# Pour aplatir l’image on utilise la méthode flatten
image_flattened=image.flatten()

# 3 et 4 a savoir faire pour le controle

#       i        i    i    i
#Ei = -p  * log(q) - p ln q
#       1        1    2    2

#[1,0]
#[0,1]  -> labael =(1) = (p1)
#                  (0)   (p2)

# softmax(x)i =         exi
#                 ex1 + ex2 + · · · + ex


x = (1.2, 0.5)

exo2probq1 = maths.exp(1.2) / (maths.exp(1.2) + maths.exp(0.5))

exo2probq2  = maths.exp(0.5) / (maths.exp(1.2) + maths.exp(0.5))



print("probq1 " ,  exo2probq1)
print("probq2 " ,  exo2probq2)

#(0.66)
#(0.33)

#il y a plus de chance que l'images nest pas diagonal

#3) p= (1)
#      (0)

entrepriecroise_exo4_0 = -1* maths.log(exo2probq1) - 0*maths.log(exo2probq2)

print()

print(entrepriecroise_exo4_0)
print()

xImage1 = [1.5, 0.2]
xImage2 = [0.3, 1.1]


exo4_1probq1_1 = maths.exp(1.5) / (maths.exp(1.5) + maths.exp(0.2))
exo4_1probq1_2  = maths.exp(0.2) / (maths.exp(0.2) + maths.exp(1.5))

print("exo4 prob1",exo4_1probq1_1,exo4_1probq1_2)

exo4_1probq2_1 = maths.exp(0.3) / (maths.exp(0.3) + maths.exp(1.1))
exo4_1probq2_2  = maths.exp(1.1) / (maths.exp(1.1) + maths.exp(0.3))

print("exo4 prob2",exo4_1probq2_1,exo4_1probq2_2)


entrepriecroise_exo4_1 = -1* maths.log(exo4_1probq1_1) - 0*maths.log(exo4_1probq1_2)
entrepriecroise_exo4_2 = -0* maths.log(exo4_1probq2_1) - 1*maths.log(exo4_1probq2_2)
print()
print(exo4_1probq1_1)
print(exo4_1probq1_2)
print()
print()
print(exo4_1probq2_1)
print(exo4_1probq2_2)
print()
print("exo4 le best",entrepriecroise_exo4_1,entrepriecroise_exo4_2)

#q = (q1, q2)