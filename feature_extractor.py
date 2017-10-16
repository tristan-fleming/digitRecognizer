import numpy as np
import scipy as sc
from scipy import misc
import matplotlib.pyplot as plt
import helpers as h

def bounding_box(coords):
    x,y = zip(*coords)
    maxX = max(x)
    minX = min(x)
    maxY = max(y)
    minY = min(y)
    aspectRatio = (maxX - minX +1) / (maxY - minY + 1)
    boundingBox = [(minX, minY), (minX, maxY), (maxX, maxY), (maxX, minY), (minX, minY)]
    xBox, yBox = zip(*boundingBox)
    blacknessRatio = len(coords)/((maxX - minX)*(maxY - minY))
    plt.scatter(x,y,marker=',')
    plt.plot(xBox, yBox, 'b-')
    plt.show()
    return blacknessRatio, aspectRatio
