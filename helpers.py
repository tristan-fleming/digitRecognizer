import matplotlib.pyplot as plt

def plot_points(arr):
    """ Plots a scatter plot from a list of tuples """
    # Makes 2 arrays of each x,y coord
    x,y = zip(*arr)

    plt.scatter(x,y,marker=',')
    plt.show()
