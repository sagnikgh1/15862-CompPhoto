import numpy as np
import skimage
import argparse

from utils import linearize_image

def parse_args():

    parser = argparse.ArgumentParser(description='15862 Assignment 1')

    parser.add_argument('--impath', default='../data/campus.tiff', help='Path to RAW image')

    parser.add_argument('--black', default=150, help='Black level')

    parser.add_argument('--white', default=4095, help='Saturation level')

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
    print(f"Top left 2x2 square: {im[:2,:2]}")

    # White balancing
    