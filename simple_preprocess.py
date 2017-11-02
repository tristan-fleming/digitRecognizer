import numpy as np
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
from scipy import misc

import image_open as io

def photo_neg(img_np):
    '''Calculates and displays a photographic negative of a numpy array of pixel
    values'''
    img_np_neg = 255 - img_np
    plt.figure()
    ax = plt.gca()
    ax.set_xlim((0, img_np.shape[1]))
    ax.set_ylim((img_np.shape[0], 0))
    plt.imshow(img_np_neg, cmap = 'gray')
    plt.show(block = False)
    return img_np_neg

def thres_binary(img_np, thres):
    '''Binarizes a grey-scale image based on a given threshold value between 0
    and 255'''
    img_np_bin = (img_np > thres)*1
    plt.figure()
    ax = plt.gca()
    ax.set_xlim((0, img_np_bin.shape[1]))
    ax.set_ylim((img_np_bin.shape[0], 0))
    plt.imshow(img_np_bin, cmap = 'gray')
    plt.show(block = False)
    return img_np_bin

def downsample(img_np, x_scale, y_scale, funct):
    '''Downsamples image by x_scale along x-axis and y_scale along y-axis
    according to the function funct'''
    block_size = (y_scale, x_scale)
    img_np_ds = block_reduce(img_np, block_size, funct)
    return img_np_ds

def run_image_preprocess(filename):
    '''Runs the above pre-processing steps on the image file "filename"'''
    img_np = io.read_image(filename) #dtype = 'uint8'
    (num_rows, num_cols) = img_np.shape
    if num_rows > 500 or num_cols > 500:
        scale = int(np.ceil(float(max(num_cols, num_rows))/500))
        img_np_ds = downsample(img_np, scale, scale, np.mean) #dtype = 'float64'
    else:
        img_np_ds = img_np
    img_np_bin = thres_binary(img_np_ds, 128) #dtype = 'int32'
    return img_np_bin
