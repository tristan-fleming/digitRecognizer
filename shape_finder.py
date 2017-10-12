import numpy as np
import scipy as sc
from scipy import misc
import matplotlib.pyplot as plt

def find_fg_points(matrix, fg_val=0):
    """Returns all "foreground points"
    of a given 2-D numpy matrix. Foreground value defaults to 1 (white), use
    a value of 0 if the slicer ouputs black."""
    fg_list = []

    for (index, point) in np.ndenumerate(matrix):
        if (point == fg_val):
            fg_list.append(index)

    return fg_list

def flood_fill(matrix,point,shape_points):
    x = point[0]
    y = point[1]
    if matrix[x][y] == 0:
        shape_points.append((x,y))
        matrix[x][y] = 1
        if x + 1 < (matrix.shape[0]):
            flood_fill(matrix, (x+1,y), shape_points)
        if x - 1 >= 0:
            flood_fill(matrix, (x-1,y), shape_points)
        if y + 1 < (matrix.shape[1]):
            flood_fill(matrix, (x, y+1), shape_points)
        if y - 1 >= 0:
            flood_fill(matrix, (x, y-1), shape_points)
    return matrix, shape_points

def find_shapes(matrix):
    fg_points = find_fg_points(matrix, 0)
    shape_points = []
    matrix, shape_points = flood_fill(matrix,fg_points[0],shape_points)
    return shape_points
