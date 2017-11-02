from scipy import misc
import matplotlib.pyplot as plt
import cPickle, gzip

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

def read_MNIST:
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set, valid_set, test_set
