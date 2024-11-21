import os
import numpy as np
import cv2
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
import argparse
from copy import deepcopy

from utils import DecomposeI, EnforceIntegrability

from cp_hw2 import lRGB2XYZ
from cp_hw5 import integrate_frankot, integrate_poisson

def parse_args():

    parser = argparse.ArgumentParser(description='15862 Assignment 5')

    parser.add_argument('--data_dir', default='../data/obj2', help='Path to data')

    args = parser.parse_args()

    return args


if __name__=="__main__":

    # Parse args
    args = parse_args()

    # Load images
    I = []
    for i in range(1,8):
        #im_avg = np.array([tiff.imread(os.path.join(args.data_dir, f'im{i}_{j}.tiff'))[1243:2206,2405:3688] for j in range(1,11)]) / (2**16 - 1)
        im_avg = np.array([tiff.imread(os.path.join(args.data_dir, f'im{i}_{j}.tiff'))[535:2232,3079:3631] for j in range(1,11)]) / (2**16 - 1)
        im_avg = np.mean(im_avg, axis=0)
        I.append(im_avg)
    I = np.array(I)

    # Extract luminance
    I = np.array([lRGB2XYZ(I[i])[:,:,1] for i in range(I.shape[0])])
    plt.imsave("../data/im_capt_2.png", I[0], cmap='gray')

    # Reshape
    _, h, w = I.shape
    I = I.reshape((I.shape[0],-1))
    
    # Estimate L and B
    L, B = DecomposeI(I)
    A = np.linalg.norm(B, axis=0)
    N = B / A[None,:]
    A = A.reshape((h,w))
    N = N.T
    N = N.reshape((h,w,3))
    plt.imsave("../data/A_pre_2.png", A, cmap='gray')

    # Enforce integrability and apply GBR transform
    B = EnforceIntegrability(B, (h,w), 50)
    G = np.array([[1,0,0],
                  [0,1,0],
                  [0,0,-1]])
    B_orig = deepcopy(B)
    B = G @ B
    A = np.linalg.norm(B, axis=0)
    N = B / A[None,:]
    A = A.reshape((h,w))
    N = N.T
    N = N.reshape((h,w,3))
    plt.imsave("../data/A_capt_2.png", A, cmap='gray')
    plt.imsave("../data/N_capt_2.png", (N+1)/2, cmap='rainbow')

    # Compute depth
    Z = integrate_poisson(-N[:,:,0]/(N[:,:,2]-1e-10), -N[:,:,1]/(N[:,:,2]-1e-10))
    Z = Z - Z.min()
    Z = Z / Z.max()
    plt.imsave("../data/Z_capt_2.png", Z, cmap='gray')
    H, W = Z.shape
    x, y = np.meshgrid(np.arange(0,W), np.arange(0,H))
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ls = LightSource()
    color_shade = ls.shade(-Z, plt.cm.gray)
    surf = ax.plot_surface(x, y, -Z, facecolors=color_shade, rstride=4, cstride=4)
    plt.axis('off')
    plt.show()

    # New direction rendering
    l = np.array([-0.1418, -0.1804, -0.9267]).T
    I_rendered = l @ B_orig
    plt.imsave("../data/I_rendered_2.png", I_rendered.reshape(h,w), cmap='gray')