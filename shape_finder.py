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

def find_shapes(matrix, fg_val):
    m = np.copy(matrix)
    fg = find_fg_points(m, fg_val)
    num_rows, num_cols = m.shape
    shapes = []
    indTest = []
    while True:
        shape_points = []
        shapesFlat = [item for sublist in shapes for item in sublist]
        indTest = [element for element in fg if element not in shapesFlat]
        print(len(indTest))
        x,y = indTest[0][0],indTest[0][1]
        shape_points = flood_fill(m, num_rows, num_cols, x, y, shape_points, fg_val)
        shapes.append(shape_points)
        if len(indTest) == 0:
            break
    return shapes

def flood_fill(data, num_rows, num_cols, row_start, col_start, shape_points, fg_val):
    stack = [(row_start, col_start)]

    while stack:
        (row, col), *stack = stack

        if data[row, col] == fg_val:
            shape_points.append((row,col))
            if fg_val == 0:
                data[row, col] = 1
            elif fg_val == 1:
                data[row, col] = 0
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
    return shape_points
