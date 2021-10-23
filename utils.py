import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from numpy.core.numeric import full
from skimage.filters import gaussian
from skimage.util import img_as_float, img_as_ubyte

verbose = None

# global filepaths
shape_dir = "data/shape_vec/"
autoshape_dir = "data/autoshape_vec/"
minmax_dir = "data/minmax/"
warp_dir = "data/warp/"
output_dir = "output/"
im_ext = ".jpg"

def printVerbose(*args):
    ''' Wrapper to print if verbose flag. '''
    if verbose:
        print(args)

def read_img(im_path, rot=False):
    ''' Read image from im_path, rotating 
    horizontally & vertically if rot. '''
    im = plt.imread(im_path)
    return np.rot90(im, k=-1, axes=(0, 1)) if rot else im

def save_img(im, im_path, im_name):
    ''' Save image im with name im_name to im_path. '''
    printVerbose(f"saving {im_name} to {im_path}")
    os.makedirs(im_path, exist_ok=True)
    plt.imsave(im_path + im_name + ".jpg", im.astype(np.uint8))

def im_to_uin8(im):
    ''' Return im as np.uint8. '''
    return img_as_ubyte(im)

def im_to_float(im):
    ''' Return im as float. '''
    return img_as_float(im)

def save_shape(shape, shape_path, shape_name):
    ''' Save shape with name shape_name to shape_path. '''
    printVerbose(f"saving {shape_name} to {shape_path}")
    os.makedirs(shape_path, exist_ok=True)
    np.savetxt(shape_path + shape_name + ".txt", shape)

def getOrNone(path, is_im=True):
    ''' Gets file from path if exists, otherwise returns None. '''
    if os.path.isfile(path):
        printVerbose(f"reading file at {path}")
        return read_img(path) if is_im else np.loadtxt(path)
    return None

def imshow(im, shape=None, save_path=None, cmap=None, s=1):
    ''' Show the image im, potentially overlaying x, y coords shape
    and saving to save_path if not None. '''
    plt.imshow(im, cmap=cmap)
    if shape is not None:
        plt.scatter(shape[:,0], shape[:,1], s=s, c='r')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def make_dir(dir):
    os.makedirs(dir, exist_ok=True)

def get_shape(im, im_name, shape_path=shape_dir, full_path=None, ext="vec"):
    ''' Load shape {im_name}_[RANSAC/vec] from shape_path
    if possible, otherwise open ginput for manual labeling. '''
    shape_fname = f"{full_path}_{ext}" if full_path \
            else f"{shape_path}{im_name}_{ext}"
    shape_head = os.path.split(shape_fname)[0]
    ext = ".txt"
    if not os.path.isfile(shape_fname + ext):
        plt.imshow(im)
        prev_inp = (0, 0)
        pts = []
        i = 0
        while True:
            x, y = plt.ginput(1)[0]
            pt = (int(x), int(y))
            if pt == prev_inp:
                break
            prev_inp = pt
            plt.plot(x, y, '-go')
            plt.text(x + 20, y, i, color='r')
            plt.draw()
            pts.append(pt)
            i += 1
        make_dir(shape_head)
        plt.clf()
        plt.savefig(f"{shape_fname}.jpg")
        shape_vec = np.array(pts)
        np.savetxt(shape_fname + ext, shape_vec, fmt="%d")
    else:
        printVerbose(f"reading shape at {shape_fname + ext}")
        shape_vec = np.loadtxt(shape_fname + ext)
    return shape_vec

def parse_shape(shape):
    ''' Returns the height and width (and color) of shape. '''
    if len(shape) > 2:
        h, w, _ = shape
        c = True
    else:
        h, w = shape
        c = False
    return (h, w), c

def corners(shape, mode='xy'):
    ''' Returns array of x, y points representing corners
    of image shape shape. '''
    (h, w), _ = parse_shape(shape)
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]])
    if mode == 'yx':
        corners = np.flip(corners, axis=0)
    return corners

def check_within(pt, shape):
    ''' Check whether pt is contained in image of shape
    shape. '''
    (h, w), _ = parse_shape(shape)
    x, y = pt
    return y >= 0 and y < h and x >= 0 and x < w

def check_contained(pt, rect):
    ''' Check whether pt is contained in rect '''
    x_min, x_max, y_min, y_max = rect
    x, y = pt
    return y >= y_min and y < y_max and x >= x_min and x < x_max

def dist2_pairwise(x, c):
    ''' Return the squared pairwise distance between two vectors'''
    return np.sqrt((x[:,0] - c[:,0])**2 + (x[:,1] - c[:, 1])**2)

def resize(im, frac, size=None):
    ''' Resize image to frac*im.size, unless size != None
    in which case resize to size. '''
    if size is not None:
        h, w = int(size[0]), int(size[1])
    else:
        h, w = int(im.shape[0] * frac), int(im.shape[1] * frac)
    return cv2.resize(im, (w, h))

def normalize_im(im):
    ''' Normalize image  '''
    return (im - np.mean(im)) / np.std(im)

def get_mask(im):
    ''' Returns a mask on im. '''
    mask_1d = im.sum(axis=2).astype(bool).astype(float)
    if len(im.shape) > 2:
        return np.repeat(mask_1d[..., np.newaxis], 3, axis=2)
    return mask_1d

def upsample_pts(pts, curr_shape, target_shape):
    ''' Convert pts (x, y) in curr_shape to target_shape. '''
    (w, h), _ = parse_shape(curr_shape)
    (t_w, t_h), _ = parse_shape(target_shape)
    pts[:,0] = pts[:,0]/h * t_h
    pts[:,1] = pts[:,1]/w * t_w
    return pts

def extract_filename(path):
    ''' Extract filename from path. '''
    h, t = os.path.split(path)
    return t.split('.')[0], h

def extract_path(full_path):
    ''' Extract path from full_path. '''
    return full_path.split('/')[:-1]

# Multiresolution Blending Helpers

def gaussian_stack(im, l=5, sigma=2):
    '''
    Returns a gaussian stack of image im with l levels
        using a gaussian of size g_size with starting sigma
        g_sigma (increasing by power of 2 each level)
    '''
    im_stack = []
    for _ in range(l):
        im = gaussian(im, sigma=sigma, multichannel=True)
        im_stack.append(im)
    return im_stack

def laplacian_stack(im, l=5, sigma=2):
    '''
    Returns a laplacian stack of image im with l levels
        using a gaussian g(g_size, g_sigma)
    '''
    g_stack = gaussian_stack(im, l=l, sigma=sigma)
    im_stack = []
    for i in range(l-1):
        im_stack.append(g_stack[i] - g_stack[i+1])
    # L_n = G_n
    im_stack.append(g_stack[-1])
    return im_stack

def multires_blend(im1, im2, mask, levels=5, sigma=2):
    '''
    Returns an multiresolution blend of images im1, im2 using mask
        Implements the algorithm described by Burt and Adleson
    '''
    LA = laplacian_stack(im1, l=levels, sigma=sigma)
    LB = laplacian_stack(im2, l=levels, sigma=sigma)
    GR = gaussian_stack(mask, l=levels, sigma=sigma)
    LS = []
    for l in range(levels):
        LS.append(GR[l]*LA[l] + (1-GR[l])*LB[l])
    return np.sum(LS, axis=0).astype(np.uint8)