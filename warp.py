import numpy as np

import utils

def computeH(im1_pts, im2_pts):
    ''' Compute a homoography between im1_pts and im2_pts, assumes 
    len(im1_pts) == len(im2_pts) & len(im1_pts) >= 4. '''
    assert len(im1_pts) == len(im2_pts), "point arrays must have same length"
    A = []
    for (x, y), (u, v) in zip(im1_pts, im2_pts):
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    A = np.array(A)
    _, _, VT = np.linalg.svd(A)
    H = np.reshape(VT[-1,:], (3, 3))
    H /= H[2][2]
    return H

def compute_shape(shape, H):
    ''' Predict the maximal output shape of image shape
    shape under the transformation H. '''
    corners_xy = utils.corners(shape)
    x_min = y_min = float("inf")
    x_max = y_max = float("-inf")
    for x, y in corners_xy:
        proj = H @ np.array([x, y, 1]).T
        proj_x, proj_y = int(proj[0] / proj[2]), int(proj[1] / proj[2])
        if proj_x < x_min:
            x_min = proj_x
        if proj_x > x_max:
            x_max = proj_x
        if proj_y < y_min:
            y_min = proj_y
        if proj_y > y_max:
            y_max = proj_y
    out_h, out_w = y_max - y_min, x_max - x_min
    shape = (out_h, out_w, 3) if len(shape)>2 else (out_h, out_w)
    return shape, (x_min, x_max, y_min, y_max)

def warp_im(im, H, onto=True):
    ''' Warps image using homography H by computing
    the inverse transform and filling in the shape. '''
    H_inv = np.linalg.inv(H)
    warp_shape, minmax = compute_shape(im.shape, H)
    (h, w), _ = utils.parse_shape(warp_shape)
    im_warped = np.zeros(warp_shape)
    for x_w in range(w):
        for y_w in range(h):
            if onto:
                proj = H_inv @ np.array([x_w + minmax[0], y_w + minmax[2], 1]).T
            else:
                proj = H_inv @ np.array([x_w, y_w, 1]).T
            proj_x, proj_y = int(proj[0] / proj[2]), int(proj[1] / proj[2])
            if utils.check_within((proj_x, proj_y), im.shape):
                im_warped[y_w, x_w] = im[proj_y, proj_x]
    return im_warped.astype(int), minmax

def loadWarp(im, im_name, H, onto=True):
    ''' Loads warp image from warp_dir if possible,
    otherwise calls warp_im() with homography H to compute. '''
    im_name = im_name + "-warped"
    warped_im = utils.getOrNone(utils.warp_dir + im_name + ".jpg")
    minmax = utils.getOrNone(utils.minmax_dir + im_name + ".txt", is_im=False)
    if warped_im is None:
        print(f"warping {im_name}")
        warped_im, minmax = warp_im(im, H, onto=onto)
        utils.save_img(warped_im, utils.warp_dir, im_name)
        np.savetxt(utils.minmax_dir + im_name + ".txt", minmax, fmt="%d")
    return warped_im, minmax

def merge_offset(x_l, y_l, x_G, y_G):
    ''' Calculate the global start_x, start_y given a local
    x_min, y_min and global x_min, y_min. '''
    return int(x_l - x_G), int(y_l - y_G)

def merge_arbitrary(im_ref, warped_ims):
    ''' Merge arbitrary number of warped images with reference
    image im_ref. '''
    (im_ref_h, im_ref_w), _ = utils.parse_shape(im_ref.shape)
    im_ref_c = utils.corners(im_ref.shape)
    x_min = y_min = float("inf")
    x_max = y_max = float("-inf")
    for _, warp_minmax in warped_ims:
        if warp_minmax[0] < x_min: x_min = warp_minmax[0]
        if warp_minmax[1] > x_max: x_max = warp_minmax[1]
        if warp_minmax[2] < y_min: y_min = warp_minmax[2]
        if warp_minmax[3] > y_max: y_max = warp_minmax[3]
    for x, y in im_ref_c:
        if x > x_max: x_max = x
        if x < x_min: x_min = x
        if y > y_max: y_max = y
        if y < y_min: y_min = y
    merge_shape = (int(y_max - y_min), int(x_max - x_min), 3)
    merged_ims = []
    # place ref image
    merged_im = np.zeros(merge_shape)
    ref_start_x, ref_start_y = int(-x_min), int(-y_min)
    merged_im[ref_start_y:ref_start_y+im_ref_h, ref_start_x:ref_start_x+im_ref_w] = im_ref
    merged_ims.append(merged_im.astype(np.uint8))
    # place warp image(s)
    for warp_im, warp_minmax in warped_ims:
        (h, w), _ = utils.parse_shape(warp_im.shape)
        merged_im = np.zeros(merge_shape)
        start_x, start_y = merge_offset(warp_minmax[0], warp_minmax[2], x_min, y_min)
        merged_im[start_y:start_y+h, start_x:start_x+w] = warp_im
        merged_ims.append(merged_im.astype(np.uint8))
    return merged_ims