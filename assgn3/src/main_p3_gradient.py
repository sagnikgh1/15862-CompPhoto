import os
import numpy as np
import skimage
import argparse
import cv2
import matplotlib.pyplot as plt
from copy import deepcopy

from utils import poissonSolver, divergence, laplacian, gradient, fuseGradientField, gradVizProcess

def parse_args():

    parser = argparse.ArgumentParser(description='15862 Assignment 3')

    parser.add_argument('--ambient_path', default='../data/gradient_captured/gradient_ambient.jpg', help='Path to ambient image')

    parser.add_argument('--flash_path', default='../data/gradient_captured/gradient_flash.jpg', help='Path to flash image')

    args = parser.parse_args()

    return args


if __name__=="__main__":
    
    # Parse args
    args = parse_args()

    # Read images
    im_ambient = cv2.imread(args.ambient_path)
    im_ambient = cv2.cvtColor(im_ambient, cv2.COLOR_BGR2RGB) / 255
    im_flash = cv2.imread(args.flash_path)
    im_flash = cv2.cvtColor(im_flash, cv2.COLOR_BGR2RGB) / 255

    # Fuse ambient and flash images
    phi_star = []
    B = np.ones_like(im_ambient[:,:,0])
    B[0,:] = 0
    B[-1,:] = 0
    B[:,0] = 0
    B[:,-1] = 0
    grad_a_viz = np.zeros((im_ambient.shape[0], im_ambient.shape[1], 2))
    grad_phi_prime_viz = np.zeros((im_ambient.shape[0], im_ambient.shape[1], 2))
    grad_phi_star_viz = np.zeros((im_ambient.shape[0], im_ambient.shape[1], 2))
    for i in range(3): # Loop over color channels
        a = im_ambient[:,:,i]
        phi_prime = im_flash[:,:,i]
        grad_a = gradient(a)
        grad_a_viz += grad_a / 3
        grad_phi_prime = gradient(phi_prime)
        grad_phi_prime_viz += grad_phi_prime / 3
        grad_phi_star = fuseGradientField(grad_a, grad_phi_prime, phi_prime, 40, 0.1)
        grad_phi_star_viz += grad_phi_star / 3
        div_phi_star = divergence(grad_phi_star)
        phi_init = deepcopy(phi_prime)
        phi_init = (phi_prime + a) / 2
        #phi_init = phi_prime
        #phi_init = np.zeros_like(phi_prime)
        phi_star.append(poissonSolver(div_phi_star, phi_init, B, phi_prime * (1 - B), 1e-3, 1000))
    phi_star = np.stack(phi_star, axis=-1)
    phi_star = np.clip(phi_star, 0, 1)
    grad_a_viz_x, grad_a_viz_y = gradVizProcess(grad_a_viz)
    grad_phi_prime_viz_x, grad_phi_prime_viz_y = gradVizProcess(grad_phi_prime_viz)
    grad_phi_star_viz_x, grad_phi_star_viz_y = gradVizProcess(grad_phi_star_viz)
    # plt.imsave("../data/grad_a_x.jpg", (grad_a_viz_x*255).astype(np.uint8))
    # plt.imsave("../data/grad_phi_prime_x.jpg", (grad_phi_prime_viz_x*255).astype(np.uint8))
    # plt.imsave("../data/grad_phi_star_x.jpg", (grad_phi_star_viz_x*255).astype(np.uint8))
    # plt.imsave("../data/grad_a_y.jpg", (grad_a_viz_y*255).astype(np.uint8))
    # plt.imsave("../data/grad_phi_prime_y.jpg", (grad_phi_prime_viz_y*255).astype(np.uint8))
    # plt.imsave("../data/grad_phi_star_y.jpg", (grad_phi_star_viz_y*255).astype(np.uint8))
    plt.imsave("../data/fused_cptr.jpg", (phi_star*255).astype(np.uint8))