import numpy as np
import scipy as sc
import skimage as sk
from skimage.measure import block_reduce
from scipy import misc
import matplotlib.pyplot as plt

def photo_neg(img_np):
    '''Calculates and displays a photographic negative of a numpy array of pixel
    values'''
    img_np_neg = 255 - img_np
    plt.figure()
    plt.imshow(img_np_neg, cmap = 'gray')
    plt.show(block = False)
    return img_np_neg

def thres_binary(img_np, thres):
    '''Binarizes a grey-scale image based on a given threshold value between 0
    and 255'''
    img_np_bin = (img_np > thres)*1
    plt.figure()
    plt.imshow(img_np_bin, cmap = 'gray')
    plt.show(block = False)
    return img_np_bin

def downsample(img_np, x_scale, y_scale, funct):
    '''Downsamples image by x_scale along x-axis and y_scale along y-axis
    according to the function funct'''
    block_size = (x_scale, y_scale)
    img_np_ds = sk.measure.block_reduce(img_np, block_size, funct)
    return img_np_ds
