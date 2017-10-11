import numpy as np
import scipy as sc
import skimage as sk
from scipy import misc
import matplotlib.pyplot as plt

from skimage import data
from skimage.filters import threshold_local

def thres_loc(img_np, loc_blocksize, loc_method):
    '''Adaptive thresholding. Threshold value is the weighted mean for the local
    neighborhood of a pixel subtracted by some constant'''
    img_np_loc = sk.filters.threshold_local(img_np, loc_blocksize, loc_method)
    plt.figure()
    plt.imshow(img_np_loc, cmap = 'gray')
    plt.show(block = False)
    return img_np_loc

    
