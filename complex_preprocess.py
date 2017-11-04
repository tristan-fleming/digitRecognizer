import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

from skimage.morphology import skeletonize
from skimage import data, img_as_bool, morphology
from skimage.util import invert
from skimage.filters import threshold_niblack

import simple_preprocess as sp

def thres_loc(img_np, loc_blocksize):
    '''Adaptive thresholding. Threshold value is the weighted mean for the local
    neighborhood of a pixel subtracted by some constant'''
    img_np_loc = threshold_niblack(img_np, loc_blocksize, k=0.2)
    #plt.figure()
    #plt.imshow(img_np_loc, cmap = 'gray')
    #plt.show(block = False)
    return img_np_loc

def find_skeleton(img_np):
    img_np = np.uint8(img_np)
    img_np = np.uint8(np.multiply(img_np, 255/np.max(img_np)))
    # perform skeletonization
    img_bool = img_as_bool(img_np)
    #invert img_bool
    img_bool = invert(img_bool)
    skeleton = skeletonize(img_bool)
    #out = morphology.medial_axis(img_bool)
    # display results
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                             sharex=True, sharey=True,
                             subplot_kw={'adjustable': 'box-forced'})

    ax = axes.ravel()

    ax[0].imshow(img_np, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('original', fontsize=20)

    ax[1].imshow(skeleton, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('skeleton', fontsize=20)

    fig.tight_layout()
    plt.show()

    skeleton = np.uint8(skeleton)
    return skeleton #dtype = boolean
