import numpy as np
from skimage import data
import matplotlib as plt
from scipy import signal
from matplotlib.pyplot import imshow, get_cmap
import matplotlib.pyplot as plt


import numpy as np
from skimage import data

import os
import sys
sys.path.append(os.path.expanduser("~/py-packages/"))


from skimage.measure import block_reduce
import matplotlib.pyplot as plt






def displayTwoBaWImages(img1, img2):
    _, axes = plt.subplots(ncols=2)
    axes[0].imshow(img1, cmap=plt.get_cmap('gray'))
    axes[1].imshow(img2, cmap=plt.get_cmap('gray'))
    
image = np.array([
    [0,0,0,0,0],
    [0,0,1,0,0],
    [0,1,1,1,0],
    [0,0,1,0,0],
    [0,0,0,0,0]
])

image = data.checkerboard()

image_normalized = (image - np.min(image)) / (np.max(image) - np.min(image))



filtre =(np.ones((3,3), np.float32)/9)


cmp = get_cmap('gray')
imshow(image, cmap=cmp, vmin=0, vmax=1)
plt.show()


kernel = filtre
print(kernel)

imgconvol = signal.convolve2d(image,
        kernel,
        mode='same',
        boundary='fill',
        fillvalue=0)




displayTwoBaWImages(image,imgconvol )

plt.show()
