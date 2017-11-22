from scipy import misc
import matplotlib.pyplot as plt
import pickle, gzip
import numpy as np
from skimage.transform import rotate

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

def MNIST_sort(MNIST_dataset, digit, subset_len):
    digit_imgs = [x for i,x in enumerate(MNIST_dataset[0][0:subset_len]) if MNIST_dataset[1][i] == digit]
    indices = [i for i,x in enumerate(MNIST_dataset[0:subset_len]) if MNIST_dataset[1][i] == digit]
    digit_imgs_np = [np.array(x).reshape(28,28) for x in digit_imgs]
    return digit_imgs_np, indices

def MNIST_add_rot(np_list_MNISTimgs, num_rot_imgs, rot_step):
    if (num_rot_imgs) % 2 != 0:
        print("Number of rotated images to be added to the MNIST data set must be even.")
        return
    if (num_rot_imgs/2)*rot_step > 45:
        print("The largest rotation angle must be less than 45 degrees.")
        return
    rots = np.linspace(-(num_rot_imgs/2)*rot_step, (num_rot_imgs/2)*rot_step, num_rot_imgs + 1)
    rots = np.delete(rots, (num_rot_imgs/2))
    np_list_MNISTimgs_w_rot = np_list_MNISTimgs[:]
    for MNISTimg in np_list_MNISTimgs:
        for rot in rots:
            rot_img = rotate(MNISTimg, rot)
            np_list_MNISTimgs_w_rot.append(rot_img)
    return np_list_MNISTimgs_w_rot
