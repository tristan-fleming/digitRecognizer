import numpy as np
import scipy as sc
from scipy import misc
import matplotlib.pyplot as plt
import helpers as h
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line, hough_ellipse)
from skimage import data, color
from skimage.draw import ellipse_perimeter

from matplotlib import cm
import edge_finder as ef
import shape_finder as sf
import simple_preprocess as sp

def bounding_box(coords):
    '''Finds the bounding box of the given coordinates, i.e. the minimum size
    box that encloses all of the given coordinates. Also returns the blackness
    and aspect ratio of the bounding box.'''
    row,col = zip(*coords)
    maxX = max(col)
    minX = min(col)
    maxY = max(row)
    minY = min(row)
    aspectRatio = (maxY - minY +1) / (maxX - minX + 1)
    boundingBox = [(minY, minX), (minY, maxX), (maxY, maxX), (maxY, minX), (minY, minX)]
    yBox, xBox = zip(*boundingBox)
    blacknessRatio = len(coords)/((maxX - minX)*(maxY - minY))
    plt.scatter(x,y,marker=',')
    plt.plot(xBox, yBox, 'b-')
    plt.show()
    col_dim = maxX - minX
    row_dim = maxY - minY
    return blacknessRatio, aspectRatio, row_dim, col_dim


def hough_line_transform(edge_pts):
    '''Calculates the Hough line transform of a list of edge points. The Hough
    line transform maps each point in the list of edge points to a curve in
    r-theta space, where r is the shortest distance from a line intersecting the
    edge point to the origin and theta is the angle of that line.'''
    edge_pts_loc = ef.coords2array(edge_pts)
    lines = probabilistic_hough_line(edge_pts_loc, threshold=10, line_length=5,
                                 line_gap=3)

    # Generating figure 2
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(edge_pts_loc, cmap=cm.gray)
    ax[0].set_title('Input image')

    ax[1].imshow(edge_pts_loc, cmap=cm.gray)
    for line in lines:
        p0, p1 = line
        ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]))
    ax[1].set_xlim((0, edge_pts_loc.shape[1]))
    ax[1].set_ylim((edge_pts_loc.shape[0], 0))
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

def lines(img_np):
    edge_pts = ef.find_edge_points(img_np,0)
    lines = hough_line_transform(edge_pts)
    notch_pairs = find_notches(lines, 4, 120)
    return lines, notch_pairs
