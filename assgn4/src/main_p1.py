import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse

from utils import LightfieldTo5dArr, CreateSubapertureMosaic, RefocusLightfield, DepthFromFocus, GenerateFocalApertureStack, ConfocalStereo

def parse_args():

    parser = argparse.ArgumentParser(description='15862 Assignment 4')

    parser.add_argument('--lf_path', default='../data/chessboard_lightfield.png', help='Path to lightfield image')

    parser.add_argument('--lenslet_size', default=16, help='Lenslet size in pixels')

    args = parser.parse_args()

    return args


if __name__=="__main__":

    # Parse args
    args = parse_args()

    # Load lightfield
    im_lf = cv2.imread(args.lf_path)
    im_lf = cv2.cvtColor(im_lf, cv2.COLOR_BGR2RGB)

    # Convert to 5D array
    im_5d = LightfieldTo5dArr(im_lf, args.lenslet_size)

    # Convert to subaperture mosaic
    im_mosaic = CreateSubapertureMosaic(im_5d)
    plt.imsave("../data/mosaic.jpg", im_mosaic)

    # Compute focal stack
    depths = np.arange(0, 1.1, 0.2)
    im_fs = np.stack([RefocusLightfield(im_5d, d) for d in depths], axis=-1)
    for i in range(im_fs.shape[-1]):
        plt.imsave(f"../data/rf_{i}.jpg", im_fs[:,:,:,i])

    # Depth from focus
    depth_map, im_aif = DepthFromFocus(im_fs, depths, 5, 5)
    plt.imsave("../data/aif.jpg", im_aif)
    plt.imsave("../data/dff.jpg", depth_map, cmap='gray')
    plt.show()

    # Focal-aperture stack and confocal stereo
    apertures = np.arange(2, 17, 2)
    depths = [-0.25, -0.1, 0, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75]
    depths = np.array(depths)
    depths = np.arange(-0.2, 1.7, 0.2)
    im_fas = GenerateFocalApertureStack(im_5d, depths, apertures)
    plt.imsave("../data/AFI1.jpg", im_fas[:,:,2,3])
    plt.imsave("../data/AFI2.jpg", im_fas[:,:,6,8])
    plt.imsave("../data/AFI3.jpg", im_fas[:,:,5,5])
    im_fas_viz = np.transpose(im_fas, (0,2,1,3,4))
    im_fas_viz = np.reshape(im_fas_viz, (im_fas_viz.shape[0] * im_fas_viz.shape[1], -1, 3))
    plt.imsave("../data/FAS.jpg", im_fas_viz)
    depth_map = ConfocalStereo(im_fas, depths)
    plt.imsave("../data/confocal.jpg", depth_map, cmap='gray')