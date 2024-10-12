import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from scipy.signal import convolve2d
from copy import deepcopy

def piecewiseBilateral(I, sigma_s, sigma_r, lmbd=0.01):
    """
    Implements the piecewise bilateral filtering algorithm
    as described in Durand and Dorsey [3].
    """
    # Define NB_SEGMENTS
    minI = np.min(I) - lmbd
    maxI = np.max(I) + lmbd
    NB_SEGMENTS = int(np.ceil((maxI - minI) / sigma_r))

    # Compute intensity stack for each color channel
    J_ls = [computeIntesityStack(I[:,:,i], sigma_s, sigma_r, NB_SEGMENTS, minI, maxI) for i in range(3)]

    # Interpolate intensity stack
    img_filt = [interpolateIntensityStack(J_ls[i], I[:,:,i], NB_SEGMENTS, minI, maxI) for i in range(3)]
    img_filt = np.stack(img_filt, axis=-1)

    return img_filt

def jointPiecewiseBilateral(I_amb, I_flash, sigma_s, sigma_r, lmbd=0.01):
    """
    Implements the joint piecewise bilateral filtering algorithm
    as described in Petschnigg et al. [6].
    """
    # Define NB_SEGMENTS
    minI = np.min(I_flash) - lmbd
    maxI = np.max(I_flash) + lmbd
    NB_SEGMENTS = int(np.ceil((maxI - minI) / sigma_r))

    # Compute intensity stack for each color channel
    J_ls = [computeIntesityStack(I_amb[:,:,i], I_flash[:,:,i], sigma_s, sigma_r, NB_SEGMENTS, minI, maxI) for i in range(3)]

    # Interpolate intensity stack
    img_filt = [interpolateIntensityStack(J_ls[i], I_flash[:,:,i], NB_SEGMENTS, minI, maxI) for i in range(3)]
    img_filt = np.stack(img_filt, axis=-1)

    return img_filt

def computeIntesityStack(I_amb, I_flash, sigma_s, sigma_r, NB_SEGMENTS, minI, maxI):
    """
    Helper function for picewiseBilateral. This is only for a single color channel.
    """
    J = [] # Output stack

    # Loop over intensities
    for j in range(NB_SEGMENTS+1):
        ij = minI + j * (maxI - minI) / NB_SEGMENTS
        G = (1 / (sigma_r * np.sqrt(2 * np.pi))) * np.exp(-(I_flash - ij)**2 / (2*sigma_r**2))
        K = cv2.GaussianBlur(G, ksize=(0,0), sigmaX=sigma_s, sigmaY=sigma_s)
        H = G * I_amb
        H = cv2.GaussianBlur(H, ksize=(0,0), sigmaX=sigma_s, sigmaY=sigma_s)
        Jj = H / K
        J.append(Jj)

    return np.array(J)

def interpolateIntensityStack(J, I, NB_SEGMENTS, minI, maxI):
    """
    Interpolates the intensity stack to compute the final image.
    """
    # Define input grid points
    points_int = [minI + i * (maxI - minI) / NB_SEGMENTS for i in range(NB_SEGMENTS+1)]
    points_y = np.arange(I.shape[0])
    points_x = np.arange(I.shape[1])

    # Define output sample coords
    samples_x = np.arange(I.shape[1])
    samples_y = np.arange(I.shape[0])
    samples_x, samples_y = np.meshgrid(samples_x, samples_y)
    samples = np.stack([I, samples_y, samples_x], axis=-1)
    samples = np.reshape(samples, (-1, 3))

    # Interpolate
    vals = interpn((points_int, points_y, points_x), J, samples)
    img_filt = np.reshape(vals, I.shape)

    return img_filt

def detailTransfer(A_NR, F, sigma_s, sigma_r):
    """
    Implements the detail transfer step.
    """

    # Bilateral filtering the flash image
    F_base = jointPiecewiseBilateral(F, F, sigma_s, sigma_r)

    # Detail transfer
    A_detail = A_NR * (F + 1e-5) / (F_base + 1e-5)

    return A_detail

def linearize(im):
    """
    Approximately linearizes given image.
    """
    im_lin = np.zeros_like(im)
    im_lin[im<=0.0404482] = im[im<=0.0404482] / 12.92
    im_lin[im>0.0404482] = ((im[im>0.0404482] + 0.055) / 1.055) ** 2.4

    return im_lin

def luminance(im_rgb):
    """
    Compute luminance from linear RGB image.
    """
    return im_rgb[:,:,0] * 0.2126 + im_rgb[:,:,1] * 0.7152 + im_rgb[:,:,2] * 0.0722

def computeMask(A, F, tau_shad):
    """
    Computes mask for specularity and shadow masking.
    """

    A_lin = linearize(A)
    F_lin = linearize(F)

    # Compute luminance
    A_lin = luminance(A_lin)
    F_lin = luminance(F_lin)

    # Normalize for ISO
    A_lin = A_lin * (200/1600)

    # Compute masks
    M_shad = (np.abs(F_lin - A_lin) < tau_shad)
    M_shad = M_shad.astype(np.uint8)
    M_spec = (F_lin > 0.9 * np.max(F_lin))
    M_spec = M_spec.astype(np.uint8)

    # Morphological operations
    M_shad = cv2.morphologyEx(M_shad, cv2.MORPH_OPEN, np.ones((5,5)).astype(np.uint8))
    M_shad = cv2.morphologyEx(M_shad, cv2.MORPH_CLOSE, np.ones((5,5)).astype(np.uint8))
    M_shad = cv2.dilate(M_shad, np.ones((25,25)).astype(np.uint8), iterations=1)
    M_spec = cv2.dilate(M_spec, np.ones((25,25)).astype(np.uint8), iterations=1)

    # Combine masks
    M = np.logical_or(M_shad, M_spec)
    plt.imsave("M.png", M*255)

    return M

def computeDiff(im1, im2):
    """
    Computes difference between 2 images for visualization.
    """

    diff = np.abs(im1 - im2)
    diff = np.mean(diff, axis=-1)
    diff = diff / np.max(diff)
    
    return diff

def gradient(I):
    """
    Computes the gradient of an image.
    Channel 1: Ix
    Channel 2: Iy
    """

    Ix = np.concatenate([I, np.zeros((I.shape[0],1))], axis=1)[:,1:] - I
    Iy = np.concatenate([I, np.zeros((1,I.shape[1]))], axis=0)[1:,:] - I

    return np.stack([Ix, Iy], axis=-1)

def divergence(gradI):
    """
    Computes the divergence of the gradient of an image.
    """

    Ix = gradI[:,:,0]
    Iy = gradI[:,:,1]

    Ixx = Ix - np.concatenate([np.zeros((Ix.shape[0],1)), Ix], axis=1)[:,:-1]
    Iyy = Iy - np.concatenate([np.zeros((1,Iy.shape[1])), Iy], axis=0)[:-1,:]

    return Ixx + Iyy

def laplacian(I):
    """
    Computes the laplacian of an image.
    """

    filter = np.array([[0,1,0],
                       [1,-4,1],
                       [0,1,0]])
    
    I_lap = convolve2d(I, filter, mode='same', boundary='fill', fillvalue=0)

    return I_lap

def poissonSolver(D, I_init, B, I_bound, eps, iters):
    """
    Implements poisson solver to integrate gradient field.
    """

    # Initialization
    I_star = B * I_init + (1 - B) * I_bound
    r = B * (D - laplacian(I_star))
    d = deepcopy(r)
    del_new = np.sum(r*r)
    n = 0

    # Optimization
    while (np.sqrt(del_new) > eps) and (n < iters):
        q = laplacian(d)
        eta = del_new / np.sum(d * q)
        I_star += eta * B * d
        r = B * (r - eta * q)
        del_old = deepcopy(del_new)
        del_new = np.sum(r*r)
        beta = del_new / del_old
        d = r + beta * d
        n += 1

    return I_star

def fuseGradientField(grad_a, grad_phi_prime, phi_prime, sigma, tau_s):
    """
    Fuses the ambient gradient field (grad_a) and the flash gradient field
    to obtain the target fused gradient field.
    """

    # Extract gradients
    grad_a_x = grad_a[:,:,0]
    grad_a_y = grad_a[:,:,1]
    grad_phi_prime_x = grad_phi_prime[:,:,0]
    grad_phi_prime_y = grad_phi_prime[:,:,1]

    # Gradient orientation coherency map
    M = np.abs(grad_phi_prime_x * grad_a_x + grad_phi_prime_y * grad_a_y)
    M = M / (np.sqrt((grad_phi_prime_x**2 + grad_phi_prime_y**2)) + np.sqrt((grad_a_x**2 + grad_a_y**2)) + 1e-5)

    # Saturation weight map
    w_s = np.tanh(sigma * (phi_prime - tau_s))
    w_s = w_s - np.min(w_s)
    w_s = w_s / np.max(w_s)

    # Compute fused gradient field
    phi_star_x = w_s * grad_a_x + (1 - w_s) * (M * grad_phi_prime_x + (1 - M) * grad_a_x)
    phi_star_y = w_s * grad_a_y + (1 - w_s) * (M * grad_phi_prime_y + (1 - M) * grad_a_y)

    return np.stack([phi_star_x, phi_star_y], axis=-1)

def gradVizProcess(grad):
    """
    Processes a 3 channel gradient image to visualize it.
    """

    grad_x, grad_y = grad[:,:,0], grad[:,:,1]
    grad_x = grad_x - np.min(grad_x)
    grad_x = grad_x / np.max(grad_x)
    grad_y = grad_y - np.min(grad_y)
    grad_y = grad_y / np.max(grad_y)
    grad_x = np.stack([grad_x]*3, axis=-1)
    grad_y = np.stack([grad_y]*3, axis=-1)
    
    return grad_x, grad_y


