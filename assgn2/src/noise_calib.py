import os
import numpy as np
import skimage
import argparse
import cv2
import tifffile as tiff
import matplotlib.pyplot as plt

from utils import *

from cp_hw2 import read_colorchecker_gm, lRGB2XYZ, XYZ2lRGB

def parse_args():

    parser = argparse.ArgumentParser(description='15862 Assignment 2')

    parser.add_argument('--ramppath', default='../data/captured_data/ramp', help='Path to exposure stack')

    parser.add_argument('--dcpath', default='../data/captured_data/dark_current', help='Path to exposure stack')

    parser.add_argument('--exp_time', default=1, help='Exposure time')

    args = parser.parse_args()

    return args

if __name__=="__main__":
    
    # Parse args
    args = parse_args()

    # Read ramps
    print("Reading ramps")
    ramps = [tiff.imread(os.path.join(args.ramppath, f"im{i+1}.tiff")) for i in range(50)]
    ramps = np.array(ramps)

    # Read dark current
    print("dark current")
    dc = [tiff.imread(os.path.join(args.dcpath, f"im{i+1}.tiff")) for i in range(50)]
    dc = np.array(dc)
    dc = np.mean(dc, axis=0, keepdims=True)

    print("Correcting")

    # Correct for dark current
    ramps = ramps[:,::,::,2]
    dc = dc[:,::,::,2]
    ramps = ramps - dc

    # Plot histogram
    plt.hist(ramps[:,1600,1800])
    plt.show()

    mean_map = np.mean(ramps, axis=0)
    var_map = (1/(ramps.shape[0]-1)) * np.sum((ramps - mean_map[None,:,:])**2, axis=0)
    rounded_mean_map = np.round(mean_map).astype(int)
    rounded_mean_map = rounded_mean_map
    mean_unique = np.unique(rounded_mean_map)
    avg_vars = []
    for mean in mean_unique:
        avg_var = np.mean(var_map[rounded_mean_map==mean])
        avg_vars.append(avg_var)
    avg_vars = np.array(avg_vars)

    # Fit line
    p = np.polyfit(mean_unique[:2000], avg_vars[:2000], deg=1)
    print(p)

    plt.plot(mean_unique, avg_vars)
    plt.plot(mean_unique, p[0]*mean_unique + p[1])
    plt.show()

