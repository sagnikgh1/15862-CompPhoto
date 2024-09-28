import os
import numpy as np
import skimage
import argparse
import cv2
import tifffile as tiff
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from utils import *

from cp_hw2 import read_colorchecker_gm, lRGB2XYZ, XYZ2lRGB, writeHDR

def parse_args():

    parser = argparse.ArgumentParser(description='15862 Assignment 2')

    parser.add_argument('--impath', default='../data/door_stack', help='Path to exposure stack')

    parser.add_argument('--num_exps', default=16, help='Number of exposures')

    parser.add_argument('--z_min', default=0.01, help='Min tolerance')

    parser.add_argument('--z_max', default=0.99, help='Max tolerance')

    parser.add_argument('--lmbd', default=1, help='Regularization weight')

    parser.add_argument('--raw_or_jpg', default='raw', help='Use RAW or JPG images?')

    parser.add_argument('--weight_func', default='photon', help='Choice of weight function - uniform|gaussian|tent|photon')

    parser.add_argument('--lin_or_log', default='log', help='Linear or logarithmic merging?')

    parser.add_argument('--tonemap_type', default='xyz', help='Choice of tonemapping - rgb|xyz')

    parser.add_argument('--downsample_factor', default=1, help='Downsample for faster debugging')

    parser.add_argument('--scale_factor', default=1, help='Scale linear HDR image')

    args = parser.parse_args()

    return args

if __name__=="__main__":
    
    # Parse args
    args = parse_args()

    # Compute exposure times
    exp_times = np.array([(1/2048)*2**k for k in range(16)])

    # Choose weight function
    if args.weight_func=='uniform':
        weight_func = weight_func_uniform
        weight_func_vect = weight_func_uniform_vect
    elif args.weight_func=='gaussian':
        weight_func = weight_func_gaussian
        weight_func_vect = weight_func_gaussian_vect
    elif args.weight_func=='tent':
        weight_func = weight_func_tent
        weight_func_vect = weight_func_tent_vect
    elif args.weight_func=='photon':
        weight_func = weight_func_photon
        weight_func_vect = weight_func_photon_vect
    else:
        raise NotImplementedError

    if args.raw_or_jpg=='raw':
        # Read exposure stack RAW
        exp_stack = [cv2.imread(os.path.join(args.impath, f"exposure{i+1}.tiff"), cv2.IMREAD_UNCHANGED) for i in range(args.num_exps)]
        exp_stack = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in exp_stack]
        exp_stack = np.array(exp_stack)/(2**16 - 1)
        exp_stack = exp_stack[:,::args.downsample_factor,::args.downsample_factor]
        exp_stack_ldr = exp_stack
        exp_stack_lin = exp_stack

    elif args.raw_or_jpg=='jpg':
        # Read exposure stack JPG
        exp_stack_ldr = [cv2.imread(os.path.join(args.impath, f"exposure{i+1}.jpg")) for i in range(args.num_exps)]
        exp_stack_ldr = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in exp_stack_ldr]
        exp_stack_ldr = np.array(exp_stack_ldr)
        exp_stack_ldr = exp_stack_ldr[:,::args.downsample_factor,::args.downsample_factor]

        # Linearize JPG exposure stack
        w_photon = (args.weight_func=='photon')
        g = estimate_cam_response(exp_stack_ldr, exp_times, weight_func, args.lmbd, args.z_min, args.z_max, w_photon)
        g = np.squeeze(g)
        plt.plot(g)
        plt.show()
        exp_stack_lin = np.exp(g[exp_stack_ldr]) / 255
        exp_stack_ldr = exp_stack_ldr / 255

    else:
        raise NotImplementedError
    
    # Merge
    if args.lin_or_log=='log':
        I_hdr = merge_exp_stack_log(exp_stack_ldr, exp_stack_lin, exp_times, weight_func_vect, args.z_min, args.z_max)
    elif args.lin_or_log=='lin':
        I_hdr = merge_exp_stack_lin(exp_stack_ldr, exp_stack_lin, exp_times, weight_func_vect, args.z_min, args.z_max)

    # Save in HDR format
    writeHDR("../data/part1.hdr", I_hdr)

    # Scale linear hdr image
    I_hdr = I_hdr * args.scale_factor

    # Color correction
    coords_ls = np.load("../data/colorch_coords.npy") / args.downsample_factor
    rgb_vals_target = read_colorchecker_gm()
    rgb_vals_target = np.stack(rgb_vals_target, axis=-1)
    rgb_vals_target = np.reshape(rgb_vals_target, (-1,3))
    I_hdr = color_correction(I_hdr, coords_ls, rgb_vals_target)

    writeHDR("../data/part2.hdr", I_hdr)

    # Tonemap
    if args.tonemap_type=='xyz':
        I_hdr = tonemap_xyz(I_hdr, 0.15, 0.95)
    elif args.tonemap_type=='rgb':
        I_hdr = tonemap_rgb(I_hdr, 0.15, 0.95)
    I_hdr = gamma_encoding(I_hdr)
    I_hdr = np.clip(I_hdr, 0, 1)
    I_hdr = (I_hdr*255).astype(np.uint8)
    I_hdr = cv2.cvtColor(I_hdr, cv2.COLOR_RGB2BGR)
    savename = f"{args.raw_or_jpg}_{args.weight_func}_{args.lin_or_log}_{args.tonemap_type}.jpg"
    cv2.imwrite(os.path.join("../data", savename), I_hdr)
