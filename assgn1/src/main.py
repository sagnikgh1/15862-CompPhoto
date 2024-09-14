import numpy as np
import skimage
import argparse
import matplotlib.pyplot as plt

from utils import linearize_image, wb_grayworld, wb_whiteworld, wb_rescale, demosaic, linear_brightening, gamma_encoding, wb_manual

def parse_args():

    parser = argparse.ArgumentParser(description='15862 Assignment 1')

    parser.add_argument('--impath', default='../data/campus.tiff', help='Path to RAW image')

    parser.add_argument('--black', default=150, help='Black level')

    parser.add_argument('--white', default=4095, help='Saturation level')

    parser.add_argument('--r_scale', default=2.394531, help='Red channel scale for white balancing')

    parser.add_argument('--g_scale', default=1.0, help='Green channel scale for white balancing')

    parser.add_argument('--b_scale', default=1.597656, help='Blue channel scale for white balancing')

    parser.add_argument('--target_brightness', default=0.25, help='Target brightness before gamma encoding')

    parser.add_argument('--quality', default=60, help='JPG quality')

    args = parser.parse_args()

    return args

if __name__=="__main__":

    # Command line args
    args = parse_args()

    # Read RAW image
    im = skimage.io.imread(args.impath)

    print(f"Image Shape: {im.shape}")
    print(f"Image datatype: {im.dtype}")

    # Linearize image
    im = linearize_image(im, args.black, args.white)

    # Identifying bayer pattern
    x=2000
    print(f"Top left 2x2 square: {im[x:x+2,x:x+2]}")

    # White balancing
    im_gw_rggb = wb_grayworld(im, 'rggb')
    im_gw_bggr = wb_grayworld(im, 'bggr')
    im_ww_rggb = wb_whiteworld(im, 'rggb')
    im_ww_bggr = wb_whiteworld(im, 'bggr')
    im_rs_rggb = wb_rescale(im, 'rggb', args.r_scale, args.g_scale, args.b_scale)
    im_rs_bggr = wb_rescale(im, 'bggr', args.r_scale, args.g_scale, args.b_scale)

    # Demosaicing
    im_gw_rggb = demosaic(im_gw_rggb, 'rggb')
    im_gw_bggr = demosaic(im_gw_bggr, 'bggr')
    im_ww_rggb = demosaic(im_ww_rggb, 'rggb')
    im_ww_bggr = demosaic(im_ww_bggr, 'bggr')
    im_rs_rggb = demosaic(im_rs_rggb, 'rggb')
    im_rs_bggr = demosaic(im_rs_bggr, 'bggr')

    # Vizualize rggb and bggr to decide
    skimage.io.imsave("../data/rggb.jpg", np.clip((im_ww_rggb*255).astype(np.uint8)*3, 0, 255))
    skimage.io.imsave("../data/bggr.jpg", np.clip((im_ww_bggr*255).astype(np.uint8)*3, 0, 255))

    # Color space correction
    M_srgb2xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
                           [0.2126729, 0.7151522, 0.0721750],
                           [0.0193339, 0.1191920, 0.9503041]])
    M_xyz2cam = np.array([[6988,-1384,-714],
                          [-5631,13410,2447],
                          [-1485,2204,7318]]) / 10000
    M_srgb2cam = M_xyz2cam @ M_srgb2xyz
    M_srgb2cam = M_srgb2cam / np.sum(M_srgb2cam, axis=1, keepdims=True)

    im_gw_rggb = (np.linalg.inv(M_srgb2cam) @ im_gw_rggb[:,:,:,None])[:,:,:,0]
    im_ww_rggb = (np.linalg.inv(M_srgb2cam) @ im_ww_rggb[:,:,:,None])[:,:,:,0]
    im_rs_rggb = (np.linalg.inv(M_srgb2cam) @ im_rs_rggb[:,:,:,None])[:,:,:,0]

    # Brightening
    im_gw_rggb = linear_brightening(im_gw_rggb, args.target_brightness)
    im_ww_rggb = linear_brightening(im_ww_rggb, args.target_brightness)
    im_rs_rggb = linear_brightening(im_rs_rggb, args.target_brightness)

    # Gamma encoding
    im_gw_rggb = gamma_encoding(im_gw_rggb)
    im_ww_rggb = gamma_encoding(im_ww_rggb)
    im_rs_rggb = gamma_encoding(im_rs_rggb)

    # Visualize the different white-balancing methods
    skimage.io.imsave("../data/gw.png", np.clip((im_gw_rggb*255).astype(np.uint8), 0, 255))
    skimage.io.imsave("../data/ww.png", np.clip((im_ww_rggb*255).astype(np.uint8), 0, 255))
    skimage.io.imsave("../data/rs.png", np.clip((im_rs_rggb*255).astype(np.uint8), 0, 255))

    # Save in png format
    skimage.io.imsave("../data/ww.png", np.clip((im_ww_rggb*255).astype(np.uint8), 0, 255))
    # Save in jpg format
    skimage.io.imsave(f"../data/ww_{args.quality}.jpg", np.clip((im_ww_rggb*255).astype(np.uint8), 0, 255), quality=args.quality)

    # Manual white-balancing
    pt1 = [2592, 894]
    pt2 = [2774, 4078]
    pt3 = [406, 4310]

    im_wb_pt1 = wb_manual(im, pt1[0], pt1[1])
    im_wb_pt2 = wb_manual(im, pt2[0], pt2[1])
    im_wb_pt3 = wb_manual(im, pt3[0], pt3[1])
    im_wb_pt1 = demosaic(im_wb_pt1, 'rggb')
    im_wb_pt2 = demosaic(im_wb_pt2, 'rggb')
    im_wb_pt3 = demosaic(im_wb_pt3, 'rggb')
    im_wb_pt1 = (np.linalg.inv(M_srgb2cam) @ im_wb_pt1[:,:,:,None])[:,:,:,0]
    im_wb_pt2 = (np.linalg.inv(M_srgb2cam) @ im_wb_pt2[:,:,:,None])[:,:,:,0]
    im_wb_pt3 = (np.linalg.inv(M_srgb2cam) @ im_wb_pt3[:,:,:,None])[:,:,:,0]
    im_wb_pt1 = linear_brightening(im_wb_pt1, args.target_brightness)
    im_wb_pt2 = linear_brightening(im_wb_pt2, args.target_brightness)
    im_wb_pt3 = linear_brightening(im_wb_pt3, args.target_brightness)
    im_wb_pt1 = gamma_encoding(im_wb_pt1)
    im_wb_pt2 = gamma_encoding(im_wb_pt2)
    im_wb_pt3 = gamma_encoding(im_wb_pt3)
    skimage.io.imsave("../data/wb_pt1.png", np.clip((im_wb_pt1*255).astype(np.uint8), 0, 255))
    skimage.io.imsave("../data/wb_pt2.png", np.clip((im_wb_pt2*255).astype(np.uint8), 0, 255))
    skimage.io.imsave("../data/wb_pt3.png", np.clip((im_wb_pt3*255).astype(np.uint8), 0, 255))
