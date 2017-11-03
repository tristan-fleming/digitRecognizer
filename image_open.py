from scipy import misc
import matplotlib.pyplot as plt
import pickle, gzip
import numpy as np

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

def read_MNIST():
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding = 'iso-8859-1')
    f.close()
    return train_set, valid_set, test_set

def MNIST_plotter(MNIST_img):
    MNIST_np = np.array(MNIST_img).reshape(28,28)
    fig = plt.figure()
    plt.imshow(MNIST_np,cmap = 'gray')
    plt.ion()
    plt.show()
    return MNIST_np

def MNIST_sort(MNIST_dataset, digit):
    digit_imgs = [x for i,x in enumerate(MNIST_dataset[0]) if MNIST_dataset[1][i] == digit]
    indices = [i for i,x in enumerate(MNIST_dataset[0]) if MNIST_dataset[1][i] == digit]
    digit_imgs_np = [np.array(x).reshape(28,28) for x in digit_imgs]
    return digit_imgs_np, indices
