import matplotlib.pyplot as plt
from scipy import misc
import numpy as np


def plot_points(arr):
    """ Plots a scatter plot from a list of tuples """
    # Makes 2 arrays of each x,y coord
    x, y = zip(*arr)
    plt.scatter(x, y, marker=',')
    plt.show()


def plot_hist(digit_list):
    """ Takes in list of lists where each sub list contains some feature
    value for the digit of its index """
    for feature_list in digit_list:
        y, binEdges = np.histogram(feature_list)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        plt.plot(bincenters, y, '-')
    plt.show()


def plot_confusion_matrices(confusion_matrices, name):
    '''Plots each confusion matrix in "confusion_matrices" to a file denoted by
    the classifier name "name"'''
    for i, confusion_mx in enumerate(confusion_matrices):
        row_sums = confusion_mx.sum(axis=1, keepdims=True)
        norm_conf_mx = confusion_mx/row_sums
        np.fill_diagonal(norm_conf_mx, 0)
        fig, ax = plt.subplots()
        ax.matshow(norm_conf_mx, cmap=plt.cm.gray)
        ax.grid(False)
        fig.savefig("classification/{0}_classifier_num{1}_confusionMx.png".format(name, i))
        plt.close()


def plot_digits(digits, nrows, ncols, file_name_str):
    f, axes = plt.subplots(nrows, ncols)
    for i in range(nrows):
        for j in range(ncols):
            axes[i, j].imshow(digits[i*ncols + j])
            axes[i, j].axis('off')
    plt.tight_layout(pad=-0.2, w_pad=-2, h_pad=-2)
    f.savefig("digits_examples_{0}.png".format(file_name_str))
