import os
import numpy as np
import skimage
import argparse
import cv2
import tifffile as tiff
import matplotlib.pyplot as plt
import time

from utils import jointPiecewiseBilateral, detailTransfer, computeMask, computeDiff

def parse_args():

    parser = argparse.ArgumentParser(description='15862 Assignment 3')

    parser.add_argument('--ambient_path', default='../data/bilateral_captured/ambient.jpg', help='Path to ambient image')

    parser.add_argument('--flash_path', default='../data/bilateral_captured/flash.jpg', help='Path to flash image')

    args = parser.parse_args()

    return args


if __name__=="__main__":
    
    # Parse args
    args = parse_args()

    # Read images
    A = plt.imread(args.ambient_path) / 255
    F = plt.imread(args.flash_path) / 255

    # Bilateral filtering on ambient image
    A_base = jointPiecewiseBilateral(A, A, 10, 0.05)
    A_base = np.clip(A_base, 0, 1)
    A_base_diff = computeDiff(A_base, A)
    plt.imsave("../data/A_base_cptr.jpg", (A_base*255).astype(np.uint8))
    plt.imsave("../data/A_base_diff_cptr.jpg", (A_base_diff*255).astype(np.uint8))

    # Joint bilateral filtering using flash image as guide
    A_NR = jointPiecewiseBilateral(A, F, 5, 0.15)
    A_NR = np.clip(A_NR, 0, 1)
    A_NR_diff = computeDiff(A_NR, A_base)
    plt.imsave("../data/A_NR_cptr.jpg", (A_NR*255).astype(np.uint8))
    plt.imsave("../data/A_NR_diff_cptr.jpg", (A_NR_diff*255).astype(np.uint8))

    # Detail transfer
    A_detail = detailTransfer(A_NR, F, 5, 0.15)
    A_detail = np.clip(A_detail, 0, 1)
    A_detail_diff = computeDiff(A_detail, A_base)
    plt.imsave("../data/A_detail_cptr.jpg", (A_detail*255).astype(np.uint8))
    plt.imsave("../data/A_detail_diff_cptr.jpg", (A_detail_diff*255).astype(np.uint8))

    # Specularity and shadow masking
    M = computeMask(A, F, 0.001)[:,:,None]
    A_final = (1 - M) * A_detail + M * A_base
    A_final = np.clip(A_detail, 0, 1)
    A_final_diff = computeDiff(A_final, A_base)
    plt.imsave("../data/A_final_cptr.jpg", (A_final*255).astype(np.uint8))
    plt.imsave("../data/A_final_diff_cptr.jpg", (A_final_diff*255).astype(np.uint8))