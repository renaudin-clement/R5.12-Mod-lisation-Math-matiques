import numpy as np
from skimage import data

import os
import sys
sys.path.append(os.path.expanduser("~/py-packages/"))


from skimage.measure import block_reduce
import matplotlib.pyplot as plt

image = data.camera()
print(f"Nombre de pixels de l'image originale : {image.size}")

plt.imshow(image, cmap=plt.get_cmap('gray'))
plt.title('Image originale')
plt.show()

pooling_sizes = [2, 3, 4,10]


for size in pooling_sizes:
    # Utilise la fonction block_reduce de skimage pour effectuer le max pooling
    pooled_image = block_reduce(image, (size, size), np.max)
    # Affiche le nombre de pixels de l'image après max pooling
    print(f"Nombre de pixels après Max Pooling {size}x{size} : {pooled_image.size}")
    # Affiche l'image résultante avec le titre correspondant
    plt.imshow(pooled_image, cmap=plt.get_cmap('gray'))
    plt.title(f'Max Pooling {size}x{size}')
    plt.show()