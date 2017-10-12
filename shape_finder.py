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
    shape_points.append((x,y))
    matrix[x][y] = 1
    if x + 1 < (matrix.shape[0]) and matrix[x + 1,y] == 0:
        flood_fill(matrix, (x+1,y), shape_points)
    if x - 1 >= 0 and matrix[x - 1,y] == 0:
        flood_fill(matrix, (x - 1,y), shape_points)
    if y + 1 < (matrix.shape[1]) and matrix[x,y + 1] == 0:
        flood_fill(matrix, (x, y + 1), shape_points)
    if y - 1 >= 0  and matrix[x,y - 1] == 0:
        flood_fill(matrix, (x, y - 1), shape_points)
    return matrix, shape_points

def find_shapes(matrix):
    m = np.copy(matrix)
    ind = np.where(m == 0)
    x,y = ind[0][0],ind[1][0]
    shape_points = []
    num_rows, num_cols = m.shape
    shape_points = flood_fill2(m, num_rows, num_cols, x, y)
    return shape_points

def flood_fill2(data, num_rows, num_cols, row_start, col_start):
    stack = [(row_start, col_start)]

    while stack:
        (row, col), *stack = stack

        if data[row, col] == 0:
            data[row, col] = 1
            if row > 0:
                stack.append((row-1, col))
                if col > 0:
                    stack.append((row-1, col-1))
                if col < (num_cols-1):
                    stack.append((row-1, col+1))
            if row < (num_rows -1):
                stack.append((row+1, col))
                if col > 0:
                    stack.append((row+1, col-1))
                if col < (num_cols-1):
                    stack.append((row+1, col+1))
            if col > 0:
                stack.append((row, col-1))
            if col < (num_cols-1):
                stack.append((row, col-1))
    return data
