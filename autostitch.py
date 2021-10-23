import os
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import corner_harris, peak_local_max

import utils
from warp import computeH

def get_harris(im, im_name):
    ''' Loads Harris pts from autoshape_dir if possible,
    otherwise calls get_harris_corners() to compute. '''
    im_name = f"{im_name}_harris"
    if not os.path.isfile(utils.autoshape_dir + im_name + ".txt"):
        h, coords = get_harris_corners(im)
        coords_xy = np.fliplr(coords.T)
        h_vals = np.apply_along_axis(lambda xy_coord: h[xy_coord[1], xy_coord[0]], 1, coords_xy)
        coords_xy = np.hstack((coords_xy, h_vals[:,np.newaxis]))
        # harris coords saved as n x 3 matrix of form [x, y, H[y, x]]
        utils.save_shape(coords_xy, utils.autoshape_dir, im_name)
    else:
        coords_xy = np.loadtxt(utils.autoshape_dir + im_name + ".txt")
    return coords_xy

def ANMS(pts, n=500):
    ''' Perform Adaptive Non-Maximal Suppression
    on features pts, returning the top n features. '''
    suppressed_pts = []
    for xc, yc, hc in pts:
        interest_pts = []
        for x, y, h in pts:
            if hc < 0.9 * h:
                interest_pts.append([x, y])
        if len(interest_pts) > 0:
            interest_pts = np.array(interest_pts)
            dists = dist2(np.array([xc, yc])[:,np.newaxis].T, interest_pts)
            suppressed_pts.append([xc, yc, np.min(dists)])
    suppressed_pts.sort(key=lambda m: m[2], reverse=True)
    return np.array(suppressed_pts[:n])

def ANMS_wrapper(harris_pts, im_name, n=500):
    ''' Loads ANMS pts from autoshape_dir if possible,
    otherwise calls ANMS() to compute. '''
    ANMS_name = f"{im_name}_ANMS"
    if not os.path.isfile(utils.autoshape_dir + ANMS_name + ".txt"):
        ANMS_pts = ANMS(harris_pts, n=n)
        utils.save_shape(ANMS_pts, utils.autoshape_dir, ANMS_name)
    # ANMS pts saved as [x, y, min_dist]
    ANMS_pts = np.loadtxt(utils.autoshape_dir + ANMS_name + ".txt")
    return ANMS_pts

def extract_descriptors(im, pts):
    ''' Returns a dict descriptors mapping points (x, y) in pts
    to descriptor vectors of length (patch_size/spacing)^2
    '''
    descriptors = {}
    patch_size = (40, 40)
    spacing = 5
    for pt in pts:
        x, y = int(pt[0]), int(pt[1])
        offset_x, offset_y = int(patch_size[1]/2), int(patch_size[0]/2)
        patch = np.copy(im[y-offset_y:y+offset_y, x-offset_x:x+offset_x])
        subsampled_patch = utils.resize(patch, None, size=tuple(t/spacing for t in patch_size))
        subsampled_patch = utils.normalize_im(subsampled_patch)
        descriptors[(x, y)] = subsampled_patch.reshape(-1, 1).T
    return descriptors

def feature_matching(desc1, desc2, thres=0.25):
    ''' Returns 2 n x 2 nd.array mapping points (x, y) in 
    desc1 with points in desc2 based on Lowe thresholding
    on the ratio between the first and second nearest neighbor
    '''
    desc1_features = []
    desc2_features = []
    for (x1, y1), d1 in desc1.items():
        dists = []
        for (x2, y2), d2 in desc2.items():
            dists.append((dist2(d1, d2)[0,0], (x2, y2)))
        dists.sort(key=lambda d: d[0])
        e1_NN, e2_NN = dists[0], dists[1]
        if e1_NN[0]/e2_NN[0] < thres:
            desc1_features.append([x1, y1])
            desc2_features.append([e1_NN[1][0], e1_NN[1][1]])
    desc1_features = np.array(desc1_features)
    desc2_features = np.array(desc2_features)
    return desc1_features, desc2_features


def RANSAC(im1_pts, im2_pts, subsample_size=4, thres=0.5):
    ''' Perform a single RANSAC iteration on im1_pts, im2_pts of subsample_size
    with threshold thres. '''
    subpts_idx = np.random.choice(range(im1_pts.shape[0]), size=subsample_size, replace=False)
    im1_subpts, im2_subpts = im1_pts[subpts_idx], im2_pts[subpts_idx]
    H = computeH(im1_subpts, im2_subpts)
    proj_im1_pts = (H @ np.hstack((im1_pts, np.ones(im1_pts.shape[0])[:,np.newaxis])).T).T
    proj_im1_pts[:, 0] = proj_im1_pts[:, 0] / proj_im1_pts[:, 2]
    proj_im1_pts[:, 1] = proj_im1_pts[:, 1] / proj_im1_pts[:, 2]
    proj_im1_pts = proj_im1_pts[:,:2]
    dists = utils.dist2_pairwise(proj_im1_pts, im2_pts)
    return im1_pts[dists < thres], im2_pts[dists < thres]

def compute_harris(im1, im2, im1_name, im2_name, visualize=False):
    ''' Compute the harris points of im1 and im2, visualizing the output
    if visualize. '''
    im1_harris = get_harris(rgb2gray(im1), im1_name)
    im2_harris = get_harris(rgb2gray(im2), im2_name)
    utils.printVerbose("im1 harris pts:", im1_harris.shape[0], "", "im2 harris pts:", im2_harris.shape[0])
    if visualize:
        utils.imshow(im1, shape=im1_harris, save_path=f"{utils.autoshape_dir}{im1_name}_harris.jpg", s=0.1)
        utils.imshow(im2, shape=im2_harris, save_path=f"{utils.autoshape_dir}{im2_name}_harris.jpg", s=0.1)
    return im1_harris, im2_harris

def compute_ANMS(im1, im2, im1_harris, im2_harris, im1_name, im2_name, visualize=False):
    ''' Compute the ANMS points of im1 and im2 given im1_harris, im2_harris, visualizing the output
    if visualize. '''
    im1_ANMS = ANMS_wrapper(im1_harris, im1_name)
    im2_ANMS = ANMS_wrapper(im2_harris, im2_name)
    if visualize:
        utils.imshow(im1, shape=im1_ANMS, save_path=f"{utils.autoshape_dir}{im1_name}_ANMS.jpg")
        utils.imshow(im2, shape=im2_ANMS, save_path=f"{utils.autoshape_dir}{im2_name}_ANMS.jpg")
    return im1_ANMS, im2_ANMS

def match_features(im1, im2, im1_pts, im2_pts, im1_name, im2_name, thres=0.4, visualize=False):
    ''' Extract features from im1 im2 with extract_descriptors() and compute matching with
    feature_matching() of threshold thres, visualizing if visualize. '''
    im1_descriptors = extract_descriptors(im1, im1_pts)
    im2_descriptors = extract_descriptors(im2, im2_pts)
    # threshold must be tuned manually to ensure adequate features for RANSAC (>= 4)
    im1_features, im2_features = feature_matching(im1_descriptors, im2_descriptors, thres=thres)
    if visualize:
        utils.imshow(im1, shape=im1_features, save_path=f"{utils.autoshape_dir}{im1_name}_lowe.jpg")
        utils.imshow(im2, shape=im2_features, save_path=f"{utils.autoshape_dir}{im2_name}_lowe.jpg")
    return im1_features, im2_features

def compute_RANSAC(im1_features, im2_features, thres=0.15, RANSAC_iter=1000, ):
    ''' Run RANSAC_iter iterations of RANSAC on im1_features, im_features with threshold thres. '''
    im1_RANSAC, im2_RANSAC = [], []
    for _ in range(RANSAC_iter):
        im1_t, im2_t = RANSAC(im1_features, im2_features, thres=thres)
        if len(im1_t) > len(im1_RANSAC):
            im1_RANSAC, im2_RANSAC = im1_t, im2_t
    return im1_RANSAC, im2_RANSAC

def get_harris_corners(im, edge_discard=20):
    """
    This function takes a b&w image and an optional amount to discard
    on the edge (default is 5 pixels), and finds all harris corners
    in the image. Harris corners near the edge are discarded and the
    coordinates of the remaining corners are returned. A 2d array (h)
    containing the h value of every pixel is also returned.

    h is the same shape as the original image, im.
    coords is 2 x n (ys, xs).
    """

    assert edge_discard >= 20

    # find harris corners
    h = corner_harris(im, method='eps', sigma=1)
    coords = peak_local_max(h, min_distance=1, indices=True)

    # discard points on edge
    edge = edge_discard  # pixels
    mask = (coords[:, 0] > edge) & \
           (coords[:, 0] < im.shape[0] - edge) & \
           (coords[:, 1] > edge) & \
           (coords[:, 1] < im.shape[1] - edge)
    coords = coords[mask].T
    return h, coords


def dist2(x, c):
    """
    dist2  Calculates squared distance between two sets of points.

    Description
    D = DIST2(X, C) takes two matrices of vectors and calculates the
    squared Euclidean distance between them.  Both matrices must be of
    the same column dimension.  If X has M rows and N columns, and C has
    L rows and N columns, then the result has M rows and L columns.  The
    I, Jth entry is the  squared distance from the Ith row of X to the
    Jth row of C.

    Adapted from code by Christopher M Bishop and Ian T Nabney.
    """
    
    ndata, dimx = x.shape
    ncenters, dimc = c.shape
    assert(dimx == dimc, 'Data dimension does not match dimension of centers')

    return (np.ones((ncenters, 1)) * np.sum((x**2).T, axis=0)).T + \
            np.ones((   ndata, 1)) * np.sum((c**2).T, axis=0)    - \
            2 * np.inner(x, c)
