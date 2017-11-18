import matplotlib.pyplot as plt
from scipy import misc
import numpy as np

def plot_points(arr):
    """ Plots a scatter plot from a list of tuples """
    # Makes 2 arrays of each x,y coord
    x,y = zip(*arr)

    plt.scatter(x,y,marker=',')
    plt.show()

def plot_hist(digit_list):
    """ Takes in list of lists where each sub list contains some feature
    value for the digit of its index """
    for feature_list in digit_list:
        y,binEdges = np.histogram(feature_list)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        plt.plot(bincenters,y,'-')
    plt.show()
