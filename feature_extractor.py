import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line, hough_ellipse)
from skimage import data, color
from skimage.draw import ellipse_perimeter

import edge_finder as ef
import shape_finder as sf
import simple_preprocess as sp
import complex_preprocess as cp
import helpers as h

def bounding_box(img_np):
    '''Finds the bounding box of the given coordinates, i.e. the minimum size
    box that encloses all of the given coordinates. Also returns the blackness
    and aspect ratio of the bounding box.'''
    fg = sf.find_fg_points(img_np, 0)
    col,row = zip(*fg)
    maxX = max(col)
    minX = min(col)
    maxY = max(row)
    minY = min(row)
    aspectRatio = (maxY - minY +1) / (maxX - minX + 1)
    boundingBox = [(minY, minX), (minY, maxX), (maxY, maxX), (maxY, minX), (minY, minX)]
    yBox, xBox = zip(*boundingBox)
    blacknessRatio = len(fg)/((maxX - minX)*(maxY - minY))
    plt.figure()
    plt.scatter(row, col, marker=',')
    plt.plot(yBox, xBox, 'b-')
    ax = plt.gca()
    ax.set_xlim((0, img_np.shape[1]))
    ax.set_ylim((img_np.shape[0], 0))
    plt.show()
    col_dim = maxX - minX
    row_dim = maxY - minY
    return blacknessRatio, aspectRatio, boundingBox

def quadrant_bounding_box(img):
    '''Finds the blackness ratio, aspect ratio, and bounding box for each of the
    four quadrants of the digit img'''
    bb = bounding_box(img)
    img_np = img[bb[2][0][1]: bb[2][1][1], bb[2][0][0]:bb[2][2][0]]
    [num_rows, num_cols] = img_np.shape
    img_np_1 = img_np[0:int(round(num_rows/2)), 0:int(round(num_cols/2))]
    img_np_2 = img_np[0:int(round(num_rows/2)), int(round(num_cols/2)):num_cols]
    img_np_3 = img_np[int(round(num_rows/2)):num_rows, 0:int(round(num_cols/2))]
    img_np_4 = img_np[int(round(num_rows/2)):num_rows, int(round(num_cols/2)):num_cols]
    bb1 = bounding_box(img_np_1)
    bb2 = bounding_box(img_np_2)
    bb3 = bounding_box(img_np_3)
    bb4 = bounding_box(img_np_4)
    return bb1, bb2, bb3, bb4

def hough_line_transform(img_np):
    '''Calculates the Hough line transform of a list of edge points edge_pts.
    The Hough line transform maps each point in the list of edge points to a
    curve in r-theta space, where r is the shortest distance from a line
    intersecting the edge point to the origin and theta is the angle of that
    line.'''
    skeleton = cp.find_skeleton(img_np)
    #edge_pts_loc = edge_pts
    lines = probabilistic_hough_line(skeleton, threshold=10, line_length=5,
                                 line_gap=3)

    # Generating figure 2
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(skeleton, cmap=cm.gray)
    ax[0].set_title('Input image')

    ax[1].imshow(skeleton, cmap=cm.gray)
    for line in lines:
        p0, p1 = line
        ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]))
    ax[1].set_xlim((0, skeleton.shape[1]))
    ax[1].set_ylim((skeleton.shape[0], 0))
    ax[1].set_title('Probabilistic Hough')

    for a in ax:
        a.set_axis_off()
        a.set_adjustable('box-forced')

    plt.tight_layout()
    plt.show()
    return lines

def coords2slope(lines):
    '''Converts the representation of the Hough transform lines from two point
    coordinates into a slope value.'''
    m = []
    for line in lines:
        p0, p1 = line
        if p1[0]-p0[0] == 0:
            if p1[1]-p0[1] > 0:
                m.append(np.inf)
            m.append(-np.inf)
        else:
            m.append((p1[1]-p0[1])/(p1[0]-p0[0]))
    return m

def distance(pt1, pt2):
    '''Finds the distance between two points'''
    dist = ((pt2[1] - pt1[1])**2 + (pt2[0] - pt1[0])**2)**(1/2)
    return dist

def neighborhood_pts(pt, other_pts, neighborhood):
    '''Finds all the points within the list other_pts that are within the
    neighborhood neighborhood of the point pt.'''
    neighbors = []
    for i,other_pt in enumerate(other_pts):
        if distance(pt, other_pt) < neighborhood:
            neighbors.append(other_pt)
    return neighbors

def find_notches(lines, threshold_dist, threshold_angle):
    '''Finds notches in an edge by searching through the set of lines outputted
    by the Hough transform function for pairs of lines that have points within
    the threshold threshold_dist of each other and that make an angle less than
    the threshold threshold_angle with each other.'''
    notch_pairs = []
    slope = coords2slope(lines)
    #line_coords_flat = [item for sublist in lines for item in sublist]
    for ind1, (line1_pt1, line1_pt2) in enumerate(lines):
        for ind2, (line2_pt1, line2_pt2) in enumerate(lines):
            if ind1 != ind2:
                if (distance(line1_pt1, line2_pt1) < threshold_dist or
                    distance(line1_pt1, line2_pt2) < threshold_dist or
                    distance(line1_pt2, line2_pt1) < threshold_dist or
                    distance(line1_pt2, line2_pt2) < threshold_dist):
                    if np.sign(slope[ind1]) != np.sign(slope[ind2]):
                        theta1 = np.degrees(np.arctan((slope[ind1]-slope[ind2])/(1 + slope[ind1]*slope[ind2])))
                        theta2 = 180 - abs(theta1)
                        if max(abs(theta1), abs(theta2)) < threshold_angle:
                            notch_pairs.append((ind1,ind2))
    return notch_pairs

def hough_ellipse_transform(edge_pts):
    '''Calculates the Hough ellipse transform of the list of edge points
    edge_pts. The Hough ellipse transform follows a similar procedure to the
    Hough line transform except instead of transforming each edge point to a
    point in r-theta space, it transforms pairs of edge points to a point in
    the 5D space defining various ellipses: yc, xc, a, b, orientation.'''
    # Perform a Hough Transform
    # The accuracy corresponds to the bin size of a major axis.
    # The value is chosen in order to get a single high accumulator.
    # The threshold eliminates low accumulators
    edge_pts_loc = ef.coords2array(edge_pts)
    edge_pts_loc = 255*edge_pts_loc
    result = hough_ellipse(edge_pts_loc, threshold=25, accuracy=10, min_size=50, max_size=None)
    result.sort(order='accumulator')

    # Estimated parameters for the ellipse
    best = list(result[-1])
    yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    orientation = best[5]

    # Draw the ellipse on the original image
    cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)

    h.plot_points(list(zip(cx,cy)))
    plt.imshow(edge_pts_loc)

    return cy, cx

def num_holes(img_np):
    '''Finds the number of holes in a digit based on the number of connected
    background and foreground regions. Assumes that the foreground is completely
    surrounded by a background region.'''
    bg_shapes = sf.find_shapes(img_np, 1)
    fg_shapes = sf.find_shapes(img_np, 0)
    num_not_shapes = 0
    if len(fg_shapes) > 1:
        print('Disconnected digit!')
    for ind, shape in enumerate(bg_shapes):
        if len(shape) < 4:
            print('Flood fill missed a couple points!')
            num_not_shapes += 1
    num_holes = len(bg_shapes) - 1 - num_not_shapes
    return num_holes

def line_features(img_np):
    '''Finds the lines that make up the edge points in the digit and the notch
    pairs along those edges.'''
    lines = hough_line_transform(img_np)
    notch_pairs = find_notches(lines, 4, 120)
    return lines, notch_pairs
