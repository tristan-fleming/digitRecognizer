import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

def find_fg_points(img_np, fg_val=1):
    """Returns all "foreground points"
    of a given 2-D numpy matrix"""
    fg_list = list()
    fg_list = [i for i,x in np.ndenumerate(img_np) if x == fg_val]

    return fg_list

def get_shapes(points, num_neighbours):
    """ Gets a list of all shape objects in the slice. """
    points = points.copy()
    shape_points_list = []

    while points:
        origin = points.copy().pop()

        shape_points = flood_fill(points, origin, num_neighbours)
        shape_points_list.append(shape_points)

        for point in shape_points:
            points.remove(point)

    return shape_points_list

def flood_fill(points_list, origin, num_neighbours, ordered=False):
    """Black and white flood fill algorithm operating on the given set of
    points at the origin with a foreground color of white by default. Finds
    the closed shape containing the origin."""

    # We use a list for stack because order matters, and sets for the shape
    # and points because we want fast (O(1)) search
    if ordered is True:
        shape = list()
    else:
        shape = set()

    # Copy so we don't actually remove points from the original list
    points = points_list.copy()

    # Initialization
    row, col = origin
    if ordered is True:
        shape.append((row, col))
    else:
        shape.add((row, col))

    points.remove((row, col))

    stack = [origin]

    # Calling all moves for 8-way floodfill
    moves = move_list(num_neighbours)

    while stack:
        for move in moves:
            if move(row, col) in points:

                row, col = move(row, col)

                points.remove((row, col))

                if ordered is True:
                    shape.append((row, col))
                else:
                    shape.add((row, col))

                stack.append((row, col))
                break

        else:
            row, col = stack.pop()

    return shape

def move_list(num_neighbours):
    """ List of moves you can make for 8 way floodfill algorithm"""
    def north(row, col): return (row + 1, col)

    def south(row, col): return (row - 1, col)

    def east(row, col): return (row, col + 1)

    def west(row, col): return (row, col - 1)

    def northeast(row, col): return (row + 1, col + 1)

    def northwest(row, col): return (row + 1, col - 1)

    def southeast(row, col): return (row - 1, col + 1)

    def southwest(row, col): return (row - 1, col - 1)

    if num_neighbours == 8:
        moves = [
            north, south, east, west, northeast, northwest, southeast, southwest
        ]
    elif num_neighbours == 4:
        moves = [
            north, south, east, west
        ]
    return moves
