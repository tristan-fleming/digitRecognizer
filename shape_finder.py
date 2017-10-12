import numpy as np
import scipy as sc
from scipy import misc
import matplotlib.pyplot as plt

def find_fg_points(matrix, fg_val=1):
    """Returns all "foreground points"
    of a given 2-D numpy matrix. Foreground value defaults to 1 (white), use
    a value of 0 if the slicer ouputs black."""
    fg_list = []

    for (index, point) in np.ndenumerate(matrix):
        if (point == fg_val):
            fg_list.append(index)

    return fg_list

def flood_fill(matrix, point):
    x = point[0]
    y = point[1]
    shape_points = []
    if matrix[x][y] == 0:
        return
    elif matrix[x][y] == 1:
        shape_points.append([x,y])
        matrix.delete([x,y])
        flood_fill(matrix, [x+1,y])
        flood_fill(matrix, [x-1,y])
        flood_fill(matrix, [x, y+1])
        flood_fill(matrix, [x, y-1])
    return matrix, shape_points

def find_shapes(matrix):
    fg_list = find_fg_points(matrix)
    shape_list = []
    [fg_list, shape_points] = flood_fill(fg_list, fg_list[0])
    shape_list.append(shape_points)
    if len(fg_list) > 0:
        [fg_list, shape_points] = flood_fill(fg_list, fg_list[0])
        shape_list.append(shape_points)
    return shape_list
