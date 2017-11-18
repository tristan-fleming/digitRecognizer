import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

import simple_preprocess as sp

def seg_watershed(img_np):
    img_np_neg = sp.photo_neg(img_np)
    local_min = peak_local_max(img_np_neg, min_distance = 4, exclude_border = 4, indices = False)
    markers = ndi.label(local_min)[0]
    labels = watershed(img_np, markers)
    plt.imshow(labels, cmap = plt.cm.spectral, interpolation = 'nearest')
    plt.show()
    return markers
