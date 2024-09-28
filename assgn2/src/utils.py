import numpy as np
import cv2
import matplotlib.pyplot as plt

from cp_hw2 import lRGB2XYZ, XYZ2lRGB

def estimate_cam_response(exp_stack, exp_times, weight_func, lmbd, z_min, z_max, w_photon=False):
    """
    Estimates the camera's non-linear response curve
    from the exposure stack.
    """
    # Downsample
    exp_stack = exp_stack[:,::200,::200]

    # Flatten spatial dims
    exp_stack = exp_stack.reshape(exp_stack.shape[0], -1)

    # Compute dimensions of final matrix
    h = exp_stack.shape[0] * exp_stack.shape[1] + 256
    w = exp_stack.shape[1] + 256

    # Initialize A and b
    A = np.zeros((h, w))
    b = np.zeros((h, 1))

    row_num = 0
    for i in range(exp_stack.shape[0]):
        for j in range(exp_stack.shape[1]):
            A[row_num, exp_stack[i,j]] = weight_func(exp_stack[i,j]/255, z_min, z_max, exp_times[i])
            A[row_num, 256 + j] = -weight_func(exp_stack[i,j]/255, z_min, z_max, exp_times[i])
            b[row_num, 0] = np.log(exp_times[i]) * weight_func(exp_stack[i,j]/255, z_min, z_max, exp_times[i])
            row_num += 1

    for i in range(256):
        if w_photon:
            A[row_num, i-1] = lmbd
            A[row_num, i] = -2*lmbd
            A[row_num, i+1] = lmbd
        else:
            A[row_num, i-1] = lmbd*weight_func(i/255, z_min, z_max)
            A[row_num, i] = -2*lmbd*weight_func(i/255, z_min, z_max)
            A[row_num, i+1] = lmbd*weight_func(i/255, z_min, z_max)
        row_num += 1

    # Least squares
    g = np.linalg.lstsq(A, b, rcond=None)[0][:256]

    return g

def weight_func_uniform(z, z_min, z_max, exp_time=None):
    """
    Implements uniform weight function.
    """

    if (z>z_min) and (z<z_max):
        return 1
    return 0

def weight_func_uniform_vect(z, z_min, z_max, exp_times):
    """
    Implements uniform weight function, vectorized.
    """

    return np.logical_and(z>z_min, z<z_max).astype(int)

def weight_func_gaussian(z, z_min, z_max, exp_time=None):
    """
    Implements gaussian weight function.
    """

    if (z>z_min) and (z<z_max):
        return np.exp(-(4/0.5**2)*(z-0.5)**2)
    return 0

def weight_func_gaussian_vect(z, z_min, z_max, exp_times):
    """
    Implements gaussian weight function, vectorized.
    """

    return np.logical_and(z>z_min, z<z_max) * np.exp(-(4/0.5**2)*(z-0.5)**2)

def weight_func_tent(z, z_min, z_max, exp_time=None):
    """
    Implements tent weight function.
    """

    if (z>z_min) and (z<z_max):
        return max(z, 1-z)
    return 0

def weight_func_tent_vect(z, z_min, z_max, exp_times):
    """
    Implements tent weight function, vectorized.
    """

    return np.logical_and(z>z_min, z<z_max) * np.max(np.stack([z, 1-z], axis=0), axis=0)

def weight_func_photon(z, z_min, z_max, exp_time):
    """
    Implements photon weight function.
    """

    if (z>z_min) and (z<z_max):
        return exp_time
    return 0
    
def weight_func_photon_vect(z, z_min, z_max, exp_times):
    """
    Implements photon weight function, vectorized.
    """

    return np.logical_and(z>z_min, z<z_max) * exp_times[:,None,None,None]

def weight_func_optimal_vect(z, z_min, z_max, exp_times):
    """
    Implements optimal weight function, vectorized.
    """

    g = np.array([6.04,1.88,4.51])[None,None,None,:]
    var_add = np.array([998.57,107.56,265.64])[None,None,None,:]

    weights = exp_times[:,None,None,None]**2 / (g * z * (2**16 - 1) + var_add)
    return np.logical_and(z>z_min, z<z_max) * weights

def merge_exp_stack_lin(exp_stack_ldr, exp_stack_lin, exp_times, weight_func, z_min, z_max):
    """
    Merges given exposure stack into HDR image in a linear fashion.
    """

    weights = weight_func(exp_stack_ldr, z_min, z_max, exp_times)
    min_mask = np.logical_not(np.sum(exp_stack_ldr > z_min, axis=0))
    max_mask = np.logical_not(np.sum(exp_stack_ldr < z_max, axis=0))
    I_hdr = np.sum(weights * exp_stack_lin / exp_times[:,None,None,None], axis=0)
    I_hdr = I_hdr / np.sum(weights, axis=0)
    I_hdr[min_mask] = 0
    I_hdr[max_mask] = 1

    return I_hdr

def merge_exp_stack_log(exp_stack_ldr, exp_stack_lin, exp_times, weight_func, z_min, z_max):
    """
    Merges given exposure stack into HDR image in a logarithmic fashion.
    """

    weights = weight_func(exp_stack_ldr, z_min, z_max, exp_times)
    min_mask = np.logical_not(np.sum(exp_stack_ldr > z_min, axis=0))
    max_mask = np.logical_not(np.sum(exp_stack_ldr < z_max, axis=0))
    I_hdr = np.sum(weights * (np.log(exp_stack_lin + 1e-4) - np.log(exp_times[:,None,None,None])), axis=0)
    I_hdr = I_hdr / np.sum(weights, axis=0)
    I_hdr = np.exp(I_hdr)
    I_hdr[min_mask] = 0
    I_hdr[max_mask] = 1

    return I_hdr

def tonemap_rgb(I_hdr, key, burn):
    """
    Implements tonemapping HDR image in RGB space.
    """

    num_pixels = I_hdr.shape[0] * I_hdr.shape[1]
    Im_hdr = np.exp((1/(3*num_pixels)) * np.sum(np.log(I_hdr + 1e-2)))
    It_hdr = (key / Im_hdr) * I_hdr
    I_white = burn * np.max(It_hdr)
    I_tm = It_hdr * (1 + It_hdr/I_white**2)
    I_tm = I_tm / (1 + It_hdr)
    return I_tm

def tonemap_xyz(I_hdr, key, burn):

    I_hdr_XYZ = lRGB2XYZ(I_hdr)
    I_hdr_x = I_hdr_XYZ[:,:,0] / np.sum(I_hdr_XYZ, axis=-1)
    I_hdr_y = I_hdr_XYZ[:,:,1] / np.sum(I_hdr_XYZ, axis=-1)
    I_hdr_Y = I_hdr_XYZ[:,:,1]

    num_pixels = I_hdr_Y.shape[0] * I_hdr_Y.shape[1]
    Im_hdr = np.exp((1/(num_pixels)) * np.sum(np.log(I_hdr_Y + 1e-2)))
    It_hdr = (key / Im_hdr) * I_hdr_Y
    I_white = burn * np.max(It_hdr)
    I_tm = It_hdr * (1 + It_hdr/I_white**2)
    I_tm = I_tm / (1 + It_hdr)

    I_hdr_Y = I_tm
    I_hdr_X = I_hdr_x * (I_hdr_Y / I_hdr_y)
    I_hdr_Z = (I_hdr_Y / I_hdr_y) - I_hdr_X - I_hdr_Y

    I_hdr_XYZ = np.stack([I_hdr_X, I_hdr_Y, I_hdr_Z], axis=-1)
    I_hdr = XYZ2lRGB(I_hdr_XYZ)

    return I_hdr

def gamma_encoding(im):
    """
    This implements the gamma-encoding step.
    """

    thresh_mask = (im <= 0.0031308)

    im[thresh_mask] = 12.92 * im[thresh_mask]

    im[np.logical_not(thresh_mask)] = 1.055 * im[np.logical_not(thresh_mask)] ** (1/2.4) - 0.055

    return im

def color_correction(I_hdr, coords_ls, rgb_vals_target):
    """
    Implements color correction and white balancing for HDR images.
    """

    # RGB values from coords list
    I_hdr = I_hdr/np.max(I_hdr)
    rgb_vals = []
    for i in range(24):
        top_left = coords_ls[i*2].astype(int)
        bot_right = coords_ls[i*2 + 1].astype(int)
        rgb_crop = I_hdr[top_left[1]:bot_right[1], top_left[0]:bot_right[0]]
        rgb_vals.append(np.mean(rgb_crop, axis=(0,1)))
    rgb_vals = np.array(rgb_vals)

    # Convert to homogenous coords
    rgb_vals_homo = np.concatenate([rgb_vals, np.ones((24,1))], axis=1)

    # Setup and solve least squares
    A = np.zeros((3*24, 12))
    b = np.reshape(rgb_vals_target, (-1,1))
    for i in range(24):
        A[3*i,:4] = rgb_vals_homo[i]
        A[3*i+1,4:8] = rgb_vals_homo[i]
        A[3*i+2,8:12] = rgb_vals_homo[i]
    transform = np.linalg.lstsq(A, b)[0]
    transform = np.reshape(transform, (3,4))

    # Apply transform
    I_hdr = np.concatenate([I_hdr, np.ones((I_hdr.shape[0], I_hdr.shape[1], 1))], axis=-1)
    I_hdr = (transform @ I_hdr[:,:,:,None])[:,:,:,0]
    I_hdr[I_hdr<0] = 0.0 # Clip negative values to 0

    # White balancing
    I_hdr[:,:,0] = I_hdr[:,:,0] * rgb_vals_target[18, 0] / rgb_vals[18, 0]
    I_hdr[:,:,1] = I_hdr[:,:,1] * rgb_vals_target[18, 1] / rgb_vals[18, 1]
    I_hdr[:,:,2] = I_hdr[:,:,2] * rgb_vals_target[18, 2] / rgb_vals[18, 2]
    return I_hdr
