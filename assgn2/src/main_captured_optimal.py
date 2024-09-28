import os
import numpy as np
import skimage
import argparse
import cv2
import tifffile as tiff
import matplotlib.pyplot as plt

from utils import *

from cp_hw2 import read_colorchecker_gm, lRGB2XYZ, XYZ2lRGB, writeHDR

def parse_args():

    parser = argparse.ArgumentParser(description='15862 Assignment 2')

    parser.add_argument('--impath', default='../data/captured_data/exp_stack', help='Path to exposure stack')

    parser.add_argument('--num_exps', default=16, help='Number of exposures')

    parser.add_argument('--z_min', default=0.03, help='Min tolerance')

    parser.add_argument('--z_max', default=0.95, help='Max tolerance')

    parser.add_argument('--dcpath', default='../data/captured_data/dark_current', help='Path to exposure stack')

    parser.add_argument('--downsample_factor', default=1, help='Downsample for faster debugging')

    args = parser.parse_args()

    return args

if __name__=="__main__":
    
    # Parse args
    args = parse_args()

    # # Compute exposure times
    exp_times = np.array([(1/2048)*2**k for k in range(16)])

    # Read dark current
    dc = [tiff.imread(os.path.join(args.dcpath, f"im{i+1}.tiff")) for i in range(50)]
    dc = np.array(dc)
    dc = np.mean(dc, axis=0, keepdims=True)
    dc = dc[:,::args.downsample_factor,::args.downsample_factor]
    dc = dc * exp_times[:,None,None,None] / (256/2048)

    # Read exposure stack RAW
    exp_stack_raw = [tiff.imread(os.path.join(args.impath, f"exposure{i+1}.tiff")) for i in range(args.num_exps)]
    exp_stack_raw = np.array(exp_stack_raw)
    exp_stack_raw = exp_stack_raw[:,::args.downsample_factor,::args.downsample_factor]
    exp_stack_raw = exp_stack_raw - dc
    exp_stack_raw[exp_stack_raw<0] = 0
    exp_stack_raw = exp_stack_raw / (2**16 - 1)
    
    # Merge
    I_hdr = merge_exp_stack_log(exp_stack_raw, exp_stack_raw, exp_times, weight_func_optimal_vect, args.z_min, args.z_max)

    writeHDR("../data/part5.hdr", I_hdr)

    # Tonemap
    I_hdr = tonemap_xyz(I_hdr, 0.2, 0.95)
    I_hdr = gamma_encoding(I_hdr)
    I_hdr = np.clip(I_hdr, 0, 1)
    I_hdr = (I_hdr*255).astype(np.uint8)
    I_hdr = cv2.cvtColor(I_hdr, cv2.COLOR_RGB2BGR)
    savename = f"optimal_merged.jpg"
    cv2.imwrite(os.path.join("../data", savename), I_hdr)
