import os
import numpy as np
import cv2
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
import argparse
import scipy

from utils import DecomposeI, EnforceIntegrability

from cp_hw2 import lRGB2XYZ
from cp_hw5 import integrate_frankot, integrate_poisson

def parse_args():

    parser = argparse.ArgumentParser(description='15862 Assignment 5')

    parser.add_argument('--data_dir', default='../data', help='Path to data')

    args = parser.parse_args()

    return args


if __name__=="__main__":

    # Parse args
    args = parse_args()

    # Load images
    I = [tiff.imread(os.path.join(args.data_dir, f'input_{i}.tif')) for i in range(1,8)]
    I = np.array(I) / (2**16 - 1)

    # Extract luminance
    I = np.array([lRGB2XYZ(I[i])[:,:,1] for i in range(I.shape[0])])

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
    plt.imsave("../data/Ae.png", A, cmap='gray')
    plt.imsave("../data/Ne.png", (N+1)/2, cmap='rainbow')

    # Reestimate using non-diagonal matrix Q
    Q = np.array([[1,0,1],
                  [0,1,0],
                  [0,0,1]])
    Lq = Q @ L
    Bq = np.linalg.inv(Q).T @ B
    Aq = np.linalg.norm(Bq, axis=0)
    Nq = Bq / Aq[None,:]
    Aq = Aq.reshape((h,w))
    Nq = Nq.T
    Nq = Nq.reshape((h,w,3))
    plt.imsave("../data/Aq.png", Aq, cmap='gray')
    plt.imsave("../data/Nq.png", (Nq+1)/2, cmap='rainbow')

    # Enforce integrability and apply GBR transform
    B = EnforceIntegrability(B, (h,w), 9)
    G = np.array([[1,0,0],
                  [0,1,0],
                  [0,0.1,-1]])
    B = G @ B
    A = np.linalg.norm(B, axis=0)
    N = B / A[None,:]
    A = A.reshape((h,w))
    N = N.T
    N = N.reshape((h,w,3))
    plt.imsave("../data/A_final.png", A, cmap='gray')
    plt.imsave("../data/N_final.png", (N+1)/2, cmap='rainbow')

    # Compute depth
    Z = integrate_poisson(-N[:,:,0]/(N[:,:,2]-1e-10), -N[:,:,1]/(N[:,:,2]-1e-10))
    Z = Z - Z.min()
    Z = Z / Z.max()
    plt.imsave("../data/Z_final.png", Z, cmap='gray')
    H, W = Z.shape
    x, y = np.meshgrid(np.arange(0,W), np.arange(0,H))
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ls = LightSource()
    color_shade = ls.shade(Z, plt.cm.gray)
    surf = ax.plot_surface(x, y, Z, facecolors=color_shade, rstride=4, cstride=4)
    plt.axis('off')
    plt.show()

    # Calibrated photometric stereo
    L = scipy.io.loadmat("../data/sources.mat")['S'].T
    A = scipy.sparse.block_diag([L.T]*I.shape[1]) # System of equations
    b = I.flatten('F')
    B = scipy.sparse.linalg.lsqr(A, b)[0]
    B = np.reshape(B, (I.shape[1],3)).T
    A = np.linalg.norm(B, axis=0)
    N = B / A[None,:]
    A = A.reshape((h,w))
    N = N.T
    N = N.reshape((h,w,3))
    plt.imsave("../data/A_calib.png", A, cmap='gray')
    plt.imsave("../data/N_calib.png", (N+1)/2, cmap='rainbow')

    # Compute depth (calibrated)
    Z = integrate_poisson(-N[:,:,0]/(N[:,:,2]-1e-10), -N[:,:,1]/(N[:,:,2]-1e-10))
    Z = Z - Z.min()
    Z = Z / Z.max()
    plt.imsave("../data/Z_calib.png", Z, cmap='gray')
    H, W = Z.shape
    x, y = np.meshgrid(np.arange(0,W), np.arange(0,H))
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ls = LightSource()
    color_shade = ls.shade(-Z, plt.cm.gray)
    surf = ax.plot_surface(x, y, -Z, facecolors=color_shade, rstride=4, cstride=4)
    plt.axis('off')
    plt.show()