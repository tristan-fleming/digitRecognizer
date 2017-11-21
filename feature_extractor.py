import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line, hough_ellipse)
from skimage import data, color
from skimage.draw import ellipse_perimeter
from skimage.feature import hog

import edge_finder as ef
import shape_finder as sf
import simple_preprocess as sp
import complex_preprocess as cp
import helpers as h


def bounding_box(img_np, bounded = False):
    '''Finds the bounding box of the given coordinates, i.e. the minimum size
    box that encloses all of the given coordinates. Also returns the blackness
    and aspect ratio of the bounding box.'''
    fg = sf.find_fg_points(img_np, 1)
    if not fg:
        br = 0
        bb = []
    else:
        if bounded == False:
            row, col = zip(*fg)
            maxX = max(col)
            minX = min(col)
            maxY = max(row)
            minY = min(row)
            br = len(fg)/((maxX - minX + 1 )*(maxY - minY + 1))
        elif bounded == True:
            maxY, maxX = img_np.shape
            minX = 0
            minY = 0
            br = len(fg)/(maxX*maxY)

        #aspectRatio = (maxY - minY +1) / (maxX - minX + 1)
        #boundingBox = [(minX, minY), (minX, maxY), (maxX, maxY), (maxX, minY), (minX, minY)]
        #yBox, xBox = zip(*boundingBox)

        #plt.figure()
        #plt.scatter(col, row, marker=',')
        #plt.plot(yBox, xBox, 'b-')
        #ax = plt.gca()
        #ax.set_xlim((0, img_np.shape[1]))
        #ax.set_ylim((img_np.shape[0], 0))
        #plt.show()
        #col_dim = maxX - minX
        #row_dim = maxY - minY
        bb = [minX, maxX, minY, maxY]
    return br, bb

def quadrant_bounding_box(img):
    '''Finds the blackness ratio, aspect ratio, and bounding box for each of the
    four quadrants of the digit img'''
    br, bb = bounding_box(img)
    img_np = img[bb[2]: bb[3], bb[0]:bb[1]]
    [num_rows, num_cols] = img_np.shape
    img_np_1 = img_np[int(round(num_rows/2)):num_rows, 0:int(round(num_cols/2))]
    img_np_2 = img_np[int(round(num_rows/2)):num_rows, int(round(num_cols/2)):num_cols]
    img_np_3 = img_np[0:int(round(num_rows/2)), 0:int(round(num_cols/2))]
    img_np_4 = img_np[0:int(round(num_rows/2)), int(round(num_cols/2)):num_cols]
    br1, bb1 = bounding_box(img_np_1, True)
    br2, bb2 = bounding_box(img_np_2, True)
    br3, bb3 = bounding_box(img_np_3, True)
    br4, bb4 = bounding_box(img_np_4, True)
    return br1, br2, br3, br4

def eighths_bounding_box(img):
    '''Finds the blackness ratio, aspect ratio, and bounding box for each of the
    four quadrants of the digit img'''
    br, bb = bounding_box(img)
    img_np = img[bb[2]: bb[3], bb[0]:bb[1]]
    [num_rows, num_cols] = img_np.shape
    img_np_1 = img_np[int(round(2*num_rows/3)):num_rows, 0:int(round(num_cols/2))]
    img_np_2 = img_np[int(round(2*num_rows/3)):num_rows, int(round(num_cols/2)):num_cols]
    img_np_3 = img_np[int(round(num_rows/3)):int(round(2*num_rows/3)), 0:int(round(num_cols/2))]
    img_np_4 = img_np[int(round(num_rows/3)):int(round(2*num_rows/3)), int(round(num_cols/2)):num_cols]
    img_np_5 = img_np[0:int(round(num_rows/3)), 0:int(round(num_cols/2))]
    img_np_6 = img_np[0:int(round(num_rows/3)), int(round(num_cols/2)):num_cols]
    br1, bb1 = bounding_box(img_np_1, True)
    br2, bb2 = bounding_box(img_np_2, True)
    br3, bb3 = bounding_box(img_np_3, True)
    br4, bb4 = bounding_box(img_np_4, True)
    br5, bb5 = bounding_box(img_np_5, True)
    br6, bb6 = bounding_box(img_np_6, True)
    return br1, br2, br3, br4, br5, br6

def hough_line_transform(img_np):
    '''Calculates the Hough line transform of a list of edge points edge_pts.
    The Hough line transform maps each point in the list of edge points to a
    curve in r-theta space, where r is the shortest distance from a line
    intersecting the edge point to the origin and theta is the angle of that
    line.'''
    if np.count_nonzero(img_np) == 0:
        lines = []
    else:
        skeleton = cp.find_skeleton(img_np)
        #edge_pts_loc = edge_pts
        lines = probabilistic_hough_line(skeleton, threshold = 1, line_length = 1, line_gap = 1)

    #Generating figure 2
    #fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    #ax = axes.ravel()

    #ax[0].imshow(skeleton, cmap=cm.gray)
    #ax[0].set_title('Input image')

    #ax[1].imshow(skeleton, cmap=cm.gray)
    #for line in lines:
    #    p0, p1 = line
    #    ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]))
    #ax[1].set_xlim((0, skeleton.shape[1]))
    #ax[1].set_ylim((skeleton.shape[0], 0))
    #ax[1].set_title('Probabilistic Hough')

    #for a in ax:
    #    a.set_axis_off()
    #    a.set_adjustable('box-forced')

    #plt.tight_layout()
    #plt.show()
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
    fg = sf.find_fg_points(img_np, 1)
    bg = sf.find_fg_points(img_np, 0)
    bg_shapes = sf.get_shapes(bg, 4)
    fg_shapes = sf.get_shapes(fg, 8)

    num_err_shapes_bg = 0
    num_err_shapes_fg = 0
    for shape in bg_shapes:
        if len(shape) < 3:
            num_err_shapes_bg = num_err_shapes_bg + 1

    for shape in fg_shapes:
        if len(shape) < 3:
            num_err_shapes_fg = num_err_shapes_fg + 1

    #if num_fg_shapes > 1:
        #print("Disconnected digit!")
    if len(bg_shapes) > 3:
        num_bg_shapes = 3
    else:
        num_bg_shapes = len(bg_shapes)

    if len(fg_shapes) > 1:
        num_fg_shapes = 1
    else:
        num_fg_shapes = len(fg_shapes)

    num_holes = (num_bg_shapes-num_err_shapes_bg) - (num_fg_shapes-num_err_shapes_fg)
    return num_holes

def line_features(img_np):
    '''Finds the lines that make up the edge points in the digit and the notch
    pairs along those edges.'''
    lines = hough_line_transform(img_np)
    num_lines = len(lines)
    if num_lines == 0:
        maxLength = 0
    else:
        maxLength = 0
        for coords in lines:
            length = np.sqrt((coords[1][1] - coords[0][1])**2 + (coords[1][0] - coords[0][0])**2)
            if length > maxLength:
                maxLength = length
        #notch_pairs = find_notches(lines, 4, 120)
    return num_lines, maxLength

def line_features_comps(img_np):
    br, bb = bounding_box(img_np)
    img_np_bounded = img_np[bb[2]: bb[3], bb[0]:bb[1]]
    [num_rows, num_cols] = img_np_bounded.shape
    num_lines_img, max_line_length_img = line_features(img_np_bounded)
    img_top = img_np_bounded[0:num_rows, int(round(num_cols/2)):num_cols]
    img_bottom = img_np_bounded[0:num_rows, 0:int(round(num_cols/2))]
    img_left = img_np_bounded[0:int(round(num_rows/2)), 0:num_cols]
    img_right = img_np_bounded[int(round(num_rows/2)):num_rows, 0:num_cols]
    num_lines_top, max_line_length_top = line_features(img_top)
    num_lines_bottom, max_line_length_bottom = line_features(img_bottom)
    num_lines_left, max_line_length_left = line_features(img_left)
    num_lines_right, max_line_length_right = line_features(img_right)
    num_lines = (num_lines_img, num_lines_top, num_lines_bottom, num_lines_left, num_lines_right)
    max_line_length = (max_line_length_img, max_line_length_top, max_line_length_bottom, max_line_length_left, max_line_length_right)
    return num_lines, max_line_length

def HOG(img_np, bounded = True, padding = 0):
    if bounded == False:
        fd = hog(img_np, orientations = 9, pixels_per_cell = (14,14), cells_per_block = (1,1), visualise = False)
    elif bounded == True:
        br, bb = bounding_box(img_np)
        while True:
            if (bb[3]-bb[2]) < 4:
                bb[2] -= 1
                if (bb[3]-bb[2]) < 4:
                    bb[3] += 1
                else:
                    break
            else:
                break

        while True:
            if (bb[3] - bb[2]) % 2 != 0:
                bb[3] += 1
            else:
                break
        while True:
            if (bb[1]-bb[0]) < 4:
                bb[0] -= 1
                if (bb[1]-bb[0]) < 4:
                    bb[1] += 1
                else:
                    break
            else:
                break

        while True:
            if (bb[1] - bb[0]) % 2 != 0:
                bb[1] += 1
            else:
                break
        bb[1] += padding
        bb[3] += padding
        img_np_bounded = img_np[bb[2]: bb[3], bb[0]:bb[1]]
        fd = hog(img_np_bounded, orientations = 9, pixels_per_cell = (img_np_bounded.shape[1]/2, img_np_bounded.shape[0]/2), cells_per_block = (1,1), visualise = False)
    return fd

def features_MNIST(np_list_imgs_MNIST):
#def features_MNIST(np_list_imgs_MNIST_thres1, np_list_imgs_MNIST_thres2):
    br, bb = map(list,zip(*[bounding_box(x) for x in np_list_imgs_MNIST]))
    #blacknessRatio = [bounding_box(x) for x in np_list_imgs_MNIST_thres1]
    holes = [num_holes(x) for x in np_list_imgs_MNIST]
    #holes = [num_holes(x) for x in np_list_imgs_MNIST_thres2]
    num_lines = [line_features_comps(x)[0] for x in np_list_imgs_MNIST]
    num_lines_img = [x[0] for x in num_lines]
    #num_lines_top = [x[1] for x in num_lines]
    #num_lines_bottom = [x[2] for x in num_lines]
    #num_lines_left = [x[3] for x in num_lines]
    #num_lines_right = [x[4] for x in num_lines]
    max_line_length = [line_features_comps(x)[1] for x in np_list_imgs_MNIST]
    max_line_length_img = [x[0] for x in max_line_length]
    max_line_length_top = [x[1] for x in max_line_length]
    max_line_length_bottom = [x[2] for x in max_line_length]
    max_line_length_left = [x[3] for x in max_line_length]
    max_line_length_right = [x[4] for x in max_line_length]
    br_s1 = [eighths_bounding_box(x)[0] for x in np_list_imgs_MNIST]
    br_s2 = [eighths_bounding_box(x)[1] for x in np_list_imgs_MNIST]
    br_s3 = [eighths_bounding_box(x)[2] for x in np_list_imgs_MNIST]
    br_s4 = [eighths_bounding_box(x)[3] for x in np_list_imgs_MNIST]
    br_s5 = [eighths_bounding_box(x)[4] for x in np_list_imgs_MNIST]
    br_s6 = [eighths_bounding_box(x)[5] for x in np_list_imgs_MNIST]
    hog_features = [HOG(x, bounded = False) for x in np_list_imgs_MNIST]
    #features = np.asarray(list(zip(br, br_s1, br_s2, br_s3, br_s4, br_s5, br_s6, holes, num_lines_img, num_lines_top, num_lines_bottom, num_lines_left, num_lines_right, max_line_length_img, max_line_length_top, max_line_length_bottom, max_line_length_left, max_line_length_right)))
    features = np.asarray(list(zip(br, br_s1, br_s2, br_s3, br_s4, br_s5, br_s6, holes, num_lines_img, max_line_length_img, max_line_length_top, max_line_length_bottom, max_line_length_left, max_line_length_right)))
    features = np.concatenate((features, hog_features), axis =1)
    # Transform features by scaling each feature to a given range
    return features
