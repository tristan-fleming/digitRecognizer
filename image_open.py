import numpy as np
import scipy as sc
from scipy import misc
import matplotlib.pyplot as plt

def read_image(filename):
    '''Reads image file, plots the image, and returns numpy array of pixel
    values'''
    img_np = misc.imread(filename, mode = 'L') #Reads the image n 8 bit pixels,
    #black and white mode
    fig = plt.figure()
    plt.imshow(img_np, cmap = 'gray')
    plt.ion()
    plt.show() #keyword 'block' overrides blocking behaviour
    return img_np
