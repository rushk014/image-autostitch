import argparse

import utils
import warp
import autostitch

def mosaic_arb(im_paths, ref_im_path, shape_path=utils.shape_dir, diff_corr=False):
    '''Creates a mosaic of im_names and ref_im_name, saving output mosaic and 
    loading shapes from shape_path, using multiple shapes (per warped image) 
    for ref_im if diff_corr == True. '''
    im_arr = []
    shape_arr = []
    ref_im = utils.im_to_uin8(utils.read_img(ref_im_path))
    for im_path in im_paths:
        im_arr.append(utils.read_img(im_path))
    ref_im_name = utils.extract_filename(ref_im_path)[0]
    im_names = [utils.extract_filename(p)[0] for p in im_paths]
    if diff_corr:
        ref_im_pts = []
    else:
        ref_im_pts = utils.get_shape(ref_im, ref_im_name, shape_path=shape_path)
    for im, im_name in zip(im_arr, im_names):
        if diff_corr:
            ref_im_pts.append(utils.get_shape(im, f"{ref_im_name}_{im_name}", shape_path=shape_path))
        shape_arr.append(utils.get_shape(im, im_name, shape_path=shape_path))
    print('visualizing warp points')
    if diff_corr:
        for im_w, im_w_pts, ref_pts in zip(im_arr, shape_arr, ref_im_pts):
            utils.imshow(im_w, shape=im_w_pts)
            utils.imshow(ref_im, shape=ref_pts)
    else:
        utils.imshow(ref_im, shape=ref_im_pts)
        for im_w, im_w_pts in zip(im_arr, shape_arr):
            utils.imshow(im_w, shape=im_w_pts)
    H_arr = []
    if diff_corr:
        for im_pts, ref_pts in zip(shape_arr, ref_im_pts):
            H_arr.append(warp.computeH(im_pts, ref_pts))
    else:
        for pts in shape_arr:
            H_arr.append(warp.computeH(pts, ref_im_pts))
    warped_im_arr = []
    for im, im_name, H in zip(im_arr, im_names, H_arr):
        warped_im_arr.append(warp.loadWarp(im, im_name, H))
    merged_ims = warp.merge_arbitrary(ref_im, warped_im_arr)
    merged_im = merged_ims[0]
    for i in range(1, len(merged_ims)):
        merged_im = utils.multires_blend(merged_im, merged_ims[i], utils.get_mask(merged_im), levels=2)
        if i == len(merged_ims) - 1:
            out_name = "-".join([ref_im_name] + im_names)
            utils.save_img(merged_im, utils.output_dir, f"{out_name}-mosaic")
            utils.imshow(merged_im)
    
def rectify(im_path):
    ''' Load and rectify im_name. '''
    im = utils.read_img(im_path, rot=True)
    im_name = utils.extract_filename(im_path)[0]
    im_pts = utils.get_shape(im, im_name)
    rect_pts = utils.corners((200, 150))
    H = warp.computeH(im_pts, rect_pts)
    warped_im, _ = warp.warp_im(im, H, onto=False)
    utils.imshow(im, shape=im_pts)
    utils.imshow(warped_im, shape=rect_pts)
    warped_im_path = f"{im_name}-rectified"
    utils.save_img(warped_im, utils.output_dir, warped_im_path)

def autostitch_multi(ref_im_path, im_paths, visualize=False):
    '''Autostitch mosaic of reference image at ref_im_path and images at im_paths, 
    visualizing intermediate feature matching outputs if visualize. '''
    ref_im_full = utils.read_img(ref_im_path)
    ims_full = []
    for im_path in im_paths:
        ims_full.append(utils.read_img(im_path))
    im_frac = 0.2
    ref_im_name = utils.extract_filename(ref_im_path)[0]
    im_names = [utils.extract_filename(p)[0] for p in im_paths]
    utils.make_dir(utils.autoshape_dir)
    # set resize frac to limit # harris pts to process (should be ~10000)
    ims = [utils.resize(im, im_frac) for im in ims_full]
    ref_im = utils.resize(ref_im_full, im_frac)
    for idx, im in enumerate(ims):
        ref_harris, im_harris = autostitch.compute_harris(ref_im, im, ref_im_name, im_names[idx], visualize=visualize)
        ref_ANMS, im_ANMS = autostitch.compute_ANMS(ref_im, im, ref_harris, im_harris, ref_im_name, 
                                                    im_names[idx], visualize=visualize)
        ref_feat, im_feat = autostitch.match_features(ref_im, im, ref_ANMS, im_ANMS, ref_im_name, im_names[idx], 
                                                      thres=0.4, visualize=True)
        ref_RANSAC, im_RANSAC = autostitch.compute_RANSAC(ref_feat, im_feat, thres=1, RANSAC_iter=1000)
        ref_RANSAC_name = f"{ref_im_name}_{im_names[idx]}_RANSAC"
        im_RANSAC_name = f"{im_names[idx]}_RANSAC"
        ref_RANSAC = utils.upsample_pts(ref_RANSAC, ref_im.shape, ref_im_full.shape)
        im_RANSAC = utils.upsample_pts(im_RANSAC, im.shape, ims_full[idx].shape)
        utils.imshow(ref_im_full, shape=ref_RANSAC, save_path=utils.autoshape_dir + ref_RANSAC_name + utils.im_ext)
        utils.imshow(ims_full[idx], shape=im_RANSAC, save_path=utils.autoshape_dir + im_RANSAC_name + utils.im_ext)
        utils.save_shape(ref_RANSAC, utils.autoshape_dir, ref_RANSAC_name)
        utils.save_shape(im_RANSAC, utils.autoshape_dir, im_RANSAC_name)
    mosaic_arb(im_paths, ref_im_path, shape_path=utils.autoshape_dir, diff_corr=True)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--ref', type=argparse.FileType('r'), 
                        help='reference image path', required=True)
    parser.add_argument('-w', '--warp', type=argparse.FileType('r'),  
                        help='warp image path(s)', nargs='+')
    parser.add_argument('-m', '--mode', choices=['rectify', 'manual_mosaic', 'autostitch'], 
                        help='choose between image rectification, manual correspondence mosaicing \
                              and autostitched mosaicing', required=True)
    parser.add_argument('-v', '--verbose', action='store_true', 
                help='log file reads/writes to stdout & visualize intermediate autostitching outputs')
    args = vars(parser.parse_args())
    verbose = args['verbose']
    ref_path = args['ref'].name
    if args['warp'] is None:
        warp_paths = []
    else:
        warp_paths = [i.name for i in args['warp']]
    if args['mode'] == 'rectify':
        assert len(warp_paths) == 0, 'image rectification only takes 1 input image'
        rectify(ref_path)
    else:
        assert len(warp_paths) >= 1, 'mosaicing requires at least 1 warp image'
        if args['mode'] == 'autostitch':
            autostitch_multi(ref_path, warp_paths, visualize=verbose)
        elif args['mode'] =='manual_mosaic':
            mosaic_arb(warp_paths, ref_path)
        

