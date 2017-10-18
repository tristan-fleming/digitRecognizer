import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

def find_edge_points(matrix, fg_val=1):
    """Returns list of all "edges" of the array (all points that
    are NOT completely surrounded by other points). Input is a list of
    coordinates of all foreground points and corresponding bit matrix."""
    edge_pts = []
    fg_pts = find_fg_points(matrix, fg_val)

    # limiting the search to all points that are precisely the foreground
    # value, so its val is insignificant.

    for (row, col) in fg_pts:
        if (    matrix[(row,col)] != matrix[(row,col-1)] or
                matrix[(row,col)] != matrix[(row,col+1)] or
                matrix[(row,col)] != matrix[(row-1,col)] or
                matrix[(row,col)] != matrix[(row+1,col)]
                ):
            edge_pts.append((row, col))

    return edge_pts

def find_fg_points(matrix, fg_val=0):
    """Returns all "foreground points"
    of a given 2-D numpy matrix. Foreground value defaults to 1 (white), use
    a value of 0 if the slicer ouputs black."""
    fg_list = []

    for (index, point) in np.ndenumerate(matrix):
        if (point == fg_val):
            fg_list.append(index)

    return fg_list

def find_list_difference(total_list, removal_list):
    """Returns a list of the set difference between the total list and removal
    list. This assumes that order doesn't matter, and that every point is unique
    otherwise, the set and list will not contain the same values. Also assumed
    is that the total list and removal list are of the same type and rank.
    """
    total_set = set(total_list)
    removal_set = set(removal_list)

    difference_list = list(total_set - removal_set)

    return difference_list


def build_selection_list(point, radius=2):
    """ Returns a list of points surrounding a given point within a size of
    size (defaults to 2). This list is used to reduce the scope in the search
    for the closest point. We use this list to limit the for loop to only
    looping through values in the intersection of all edge points and this area
    around the point in question."""
    xcomp, ycomp = point[0], point[1]

    xrange = range(xcomp - radius, (xcomp+1) + radius)
    yrange = range(ycomp - radius, (ycomp+1) + radius)

    selection_list = [(i,j) for i in xrange for j in yrange]

    return selection_list


def find_closest_point(origin, pts_list, removal_pts=[]):
    """ Compares distance between point and all points in the array, and returns
    the point in that is closest as a numpy array same. Current list defaults
    to empty, so the search is throughout the whole list. Returns a tuple."""
    first_run = True

    difference_list = find_list_difference(pts_list, removal_pts)

    radius = build_selection_list(origin)
    selection_list = [point for point in difference_list if point in radius]

    selection_array = np.array(selection_list)
    origin_array = np.array(origin)

    for point in selection_array:
        dist = np.linalg.norm(point - origin_array)

        # The same point might be in the list, and we don't want to include it
        if (dist == 0):
            continue

        # Allows for the lowest_dist to be initialized to the first dist
        if first_run == True:
            first_run = False
            lowest_dist = dist
            closest_point = point

        if dist < lowest_dist:
            lowest_dist = dist
            closest_point = point

    # Must be in tuple format so we don't get a type mismatch later down the
    # road
    closest_point = tuple(closest_point.tolist())
    return closest_point


def build_edge(edge_pts):
    """ Returns a single arbitrary edge from a given edge points list """
    edge_list = []

    init = edge_pts[0] # just take the first point in the list arbitrarily
    edge_list.append(init)

    current = find_closest_point(init, edge_pts, removal_pts=edge_list)
    edge_list.append(current)

    for index, point in enumerate(edge_pts):
        next = find_closest_point(current, edge_pts, removal_pts=edge_list)
        edge_list.append(next)
        current = next

        # Remove the initial point from the edge list to ensure that on the
        # second loop find_closest_point doesn't return the initial point
        if index == 0:
            del edge_list[0]

        if (current[0] == init[0] and current[1] == init[1]):
            edge_list.append(init)

            return edge_list
    return 'Initial point not found'


def build_edge_list(edge_pts):
    """ Returns a list of all the edges of a given shape. Uses a recursive call
    to loop through until no more points are in edge_pts."""
    # Must use tuple so there is no hash errors when turning to a set type in
    # find_list_difference below
    edge_list = []

    while(True):
        edge = build_edge(edge_pts)
        edge_tuple = tuple(edge)

        edge_list.append(edge_tuple)

        flat_edge_list = [val for sublist in edge_list for val in sublist]
        edge_pts = [points for points in edge_pts if points not in flat_edge_list]

        # Find new list using a set difference
        delta = find_list_difference(edge_pts, flat_edge_list)
        print('Length delta: {}'.format(len(delta)))

        if len(delta)==0:
            return edge_list

def coords2array(coords):
    """Returns a 2D array with values of 1 at the specified edge points and
    values of 0 everywhere else"""
    row, col = zip(*coords)
    minX = min(col)
    maxX = max(col)
    minY = min(row)
    maxY = max(row)
    row_zeroed = [x - min(row) for x in row]
    col_zeroed = [y - min(col) for y in col]
    coords_zeroed = list(zip(row_zeroed, col_zeroed))
    coords_array = np.full(((maxY - minY + 1),(maxX - minX + 1)),0)
    for i,(row,col) in enumerate(coords_zeroed):
        coords_array[row][col] = 1
    return coords_array
