import numpy as np
from scipy.interpolate import RegularGridInterpolator
from skimage.color import rgb2gray

def linearize_image(im, black, white):
    """Linearize image given the black and white vals."""

    im = im - black
    im = im / white
    im = np.clip(im, 0, 1)

    return im

def wb_grayworld(im, pattern):
    """
    White balance given image using gray-world assumption.
    
    Note that this is only implemented for grbg and gbrg as
    those are the only 2 possibilities.
    """

    if pattern=='grbg':
        # Compute means of each channel
        red_mean = np.mean(im[::2,1::2]) # Red
        green_mean = (np.mean(im[::2,::2]) + np.mean(im[1::2,1::2])) / 2 # Green
        blue_mean = np.mean(im[1::2,::2]) # Blue

        # Enforce gray world assumption
        im[::2,1::2] = im[::2,1::2] / red_mean
        im[::2,::2] = im[::2,::2] / green_mean
        im[1::2,1::2] = im[1::2,1::2] / green_mean
        im[1::2,::2] = im[1::2,::2] / blue_mean

        im = im / np.max(im)

    elif pattern=='gbrg':
        # Compute means of each channel
        blue_mean = np.mean(im[::2,1::2]) # Blue
        green_mean = (np.mean(im[::2,::2]) + np.mean(im[1::2,1::2])) / 2 # Green
        red_mean = np.mean(im[1::2,::2]) # Red
        # Enforce gray world assumption
        im[::2,1::2] = im[::2,1::2] / blue_mean
        im[::2,::2] = im[::2,::2] / green_mean
        im[1::2,1::2] = im[1::2,1::2] / green_mean
        im[1::2,::2] = im[1::2,::2] / red_mean

        im = im / np.max(im)

    elif pattern=='rggb':
        # Compute means of each channel
        red_mean = np.mean(im[::2,::2]) # Red
        green_mean = (np.mean(im[::2,1::2]) + np.mean(im[1::2,::2])) / 2 # Green
        blue_mean = np.mean(im[1::2,1::2]) # Blue

        # Enforce gray world assumption
        im[::2,::2] = im[::2,::2] / red_mean
        im[::2,1::2] = im[::2,1::2] / green_mean
        im[1::2,::2] = im[1::2,::2] / green_mean
        im[1::2,1::2] = im[1::2,1::2] / blue_mean

        im = im / np.max(im)

    elif pattern=='bggr':
        # Compute means of each channel
        blue_mean = np.mean(im[::2,::2]) # Blue
        green_mean = (np.mean(im[::2,1::2]) + np.mean(im[1::2,::2])) / 2 # Green
        red_mean = np.mean(im[1::2,1::2]) # Red

        # Enforce gray world assumption
        im[::2,::2] = im[::2,::2] / blue_mean
        im[::2,1::2] = im[::2,1::2] / green_mean
        im[1::2,::2] = im[1::2,::2] / green_mean
        im[1::2,1::2] = im[1::2,1::2] / red_mean

        im = im / np.max(im)

    else:
        raise NotImplementedError
    
    return im

def wb_whiteworld(im, pattern):
    """
    White balance given image using white-world assumption.
    
    Note that this is only implemented for grbg and gbrg as
    those are the only 2 possibilities.
    """
    # Find brightest point in scene
    im_brightness = im[::2,1::2] / 3 + im[::2,::2] / 6 + im[1::2,1::2] / 6 + im[1::2,::2] / 3
    tgt_idx = 2*np.unravel_index(np.argmax(im_brightness), im_brightness.shape)
    tgt_grid = im[tgt_idx[0]:tgt_idx[0]+2,tgt_idx[1]:tgt_idx[1]+2]

    if pattern=='grbg':
        red_val = tgt_grid[0,1]
        green_val = (tgt_grid[0,0] + tgt_grid[1,1]) / 2
        blue_val = tgt_grid[1,0]
        # Enforce white world assumption
        im[::2,1::2] = im[::2,1::2] / red_val
        im[::2,::2] = im[::2,::2] / green_val
        im[1::2,1::2] = im[1::2,1::2] / green_val
        im[1::2,::2] = im[1::2,::2] / blue_val

        im = im / np.max(im)

    elif pattern=='gbrg':
        red_val = tgt_grid[1,0]
        green_val = (tgt_grid[0,0] + tgt_grid[1,1]) / 2
        blue_val = tgt_grid[0,1]
        # Enforce white world assumption
        im[::2,1::2] = im[::2,1::2] / blue_val
        im[::2,::2] = im[::2,::2] / green_val
        im[1::2,1::2] = im[1::2,1::2] / green_val
        im[1::2,::2] = im[1::2,::2] / red_val

        im = im / np.max(im)

    elif pattern=='rggb':
        red_val = tgt_grid[0,0]
        green_val = (tgt_grid[0,1] + tgt_grid[1,0]) / 2
        blue_val = tgt_grid[1,1]

        # Enforce white world assumption
        im[::2,::2] = im[::2,::2] / red_val
        im[::2,1::2] = im[::2,1::2] / green_val
        im[1::2,::2] = im[1::2,::2] / green_val
        im[1::2,1::2] = im[1::2,1::2] / blue_val

        im = im / np.max(im)

    elif pattern=='bggr':
        red_val = tgt_grid[1,1]
        green_val = (tgt_grid[0,1] + tgt_grid[1,0]) / 2
        blue_val = tgt_grid[0,0]

        # Enforce white world assumption
        im[::2,::2] = im[::2,::2] / blue_val
        im[::2,1::2] = im[::2,1::2] / green_val
        im[1::2,::2] = im[1::2,::2] / green_val
        im[1::2,1::2] = im[1::2,1::2] / red_val

        im = im / np.max(im)

    else:
        raise NotImplementedError
    
    return im

def wb_rescale(im, pattern, r_scale, g_scale, b_scale):
    """
    White balance given image using the scale factors from dcraw.
    
    Note that this is only implemented for grbg and gbrg as
    those are the only 2 possibilities.
    """

    if pattern=='grbg':
        # Rescale
        im[::2,1::2] = im[::2,1::2] * r_scale
        im[::2,::2] = im[::2,::2] * g_scale
        im[1::2,1::2] = im[1::2,1::2] * g_scale
        im[1::2,::2] = im[1::2,::2] * b_scale

        im = im / np.max(im)

    elif pattern=='gbrg':
        # Rescale
        im[::2,1::2] = im[::2,1::2] * b_scale
        im[::2,::2] = im[::2,::2] * g_scale
        im[1::2,1::2] = im[1::2,1::2] * g_scale
        im[1::2,::2] = im[1::2,::2] * r_scale

        im = im / np.max(im)

    elif pattern=='rggb':
        # Rescale
        im[::2,::2] = im[::2,::2] * r_scale
        im[::2,1::2] = im[::2,1::2] * g_scale
        im[1::2,::2] = im[1::2,::2] * g_scale
        im[1::2,1::2] = im[1::2,1::2] * b_scale

        im = im / np.max(im)

    elif pattern=='bggr':
        # Rescale
        im[::2,::2] = im[::2,::2] * b_scale
        im[::2,1::2] = im[::2,1::2] * g_scale
        im[1::2,::2] = im[1::2,::2] * g_scale
        im[::2,::2] = im[::2,::2] * r_scale

        im = im / np.max(im)

    else:
        raise NotImplementedError
    
    return im

def demosaic(im, pattern):
    """
    Demosaic given RAW image based on the pattern.

    Note that this is only implemented for grbg and gbrg as
    those are the only 2 possibilities.
    """

    # Define meshgrid
    X = np.arange(im.shape[1])
    Y = np.arange(im.shape[0])
    X, Y = np.meshgrid(X, Y, indexing='xy')

    if pattern=='grbg':
        # Interpolate red channel
        x = np.arange(1,im.shape[1],2)
        y = np.arange(0,im.shape[0],2)
        interp_red = RegularGridInterpolator((y, x), im[::2,1::2], bounds_error=False, fill_value=None)
        im_red = interp_red((Y,X))

        # Interpolate green channel
        x = np.arange(0,im.shape[1],2)
        y = np.arange(0,im.shape[0],2)
        interp_green1 = RegularGridInterpolator((y, x), im[::2,::2], bounds_error=False, fill_value=None)
        im_green1 = interp_green1((Y,X))

        x = np.arange(1,im.shape[1],2)
        y = np.arange(1,im.shape[0],2)
        interp_green2 = RegularGridInterpolator((y, x), im[1::2,1::2], bounds_error=False, fill_value=None)
        im_green2 = interp_green2((Y,X))

        im_green = (im_green1 + im_green2) / 2

        # Interpolate blue channel
        x = np.arange(0,im.shape[1],2)
        y = np.arange(1,im.shape[0],2)
        interp_blue = RegularGridInterpolator((y, x), im[1::2,::2], bounds_error=False, fill_value=None)
        im_blue = interp_blue((Y,X))

    elif pattern=='gbrg':
        # Interpolate red channel
        x = np.arange(0,im.shape[1],2)
        y = np.arange(1,im.shape[0],2)
        interp_red = RegularGridInterpolator((y, x), im[1::2,::2], bounds_error=False, fill_value=None)
        im_red = interp_red((Y,X))

        # Interpolate green channel
        x = np.arange(0,im.shape[1],2)
        y = np.arange(0,im.shape[0],2)
        interp_green1 = RegularGridInterpolator((y, x), im[::2,::2], bounds_error=False, fill_value=None)
        im_green1 = interp_green1((Y,X))

        x = np.arange(1,im.shape[1],2)
        y = np.arange(1,im.shape[0],2)
        interp_green2 = RegularGridInterpolator((y, x), im[1::2,1::2], bounds_error=False, fill_value=None)
        im_green2 = interp_green2((Y,X))

        im_green = (im_green1 + im_green2) / 2

        # Interpolate blue channel
        x = np.arange(1,im.shape[1],2)
        y = np.arange(0,im.shape[0],2)
        interp_blue = RegularGridInterpolator((y, x), im[::2,1::2], bounds_error=False, fill_value=None)
        im_blue = interp_blue((Y,X))

    elif pattern=='rggb':
        # Interpolate red channel
        x = np.arange(0,im.shape[1],2)
        y = np.arange(0,im.shape[0],2)
        interp_red = RegularGridInterpolator((y, x), im[::2,::2], bounds_error=False, fill_value=None)
        im_red = interp_red((Y,X))

        # Interpolate green channel
        x = np.arange(0,im.shape[1],2)
        y = np.arange(1,im.shape[0],2)
        interp_green1 = RegularGridInterpolator((y, x), im[::2,1::2], bounds_error=False, fill_value=None)
        im_green1 = interp_green1((Y,X))

        x = np.arange(1,im.shape[1],2)
        y = np.arange(0,im.shape[0],2)
        interp_green2 = RegularGridInterpolator((y, x), im[1::2,::2], bounds_error=False, fill_value=None)
        im_green2 = interp_green2((Y,X))

        im_green = (im_green1 + im_green2) / 2

        # Interpolate blue channel
        x = np.arange(1,im.shape[1],2)
        y = np.arange(1,im.shape[0],2)
        interp_blue = RegularGridInterpolator((y, x), im[1::2,1::2], bounds_error=False, fill_value=None)
        im_blue = interp_blue((Y,X))

    elif pattern=='bggr':
        # Interpolate red channel
        x = np.arange(1,im.shape[1],2)
        y = np.arange(1,im.shape[0],2)
        interp_red = RegularGridInterpolator((y, x), im[1::2,1::2], bounds_error=False, fill_value=None)
        im_red = interp_red((Y,X))

        # Interpolate green channel
        x = np.arange(0,im.shape[1],2)
        y = np.arange(1,im.shape[0],2)
        interp_green1 = RegularGridInterpolator((y, x), im[::2,1::2], bounds_error=False, fill_value=None)
        im_green1 = interp_green1((Y,X))

        x = np.arange(1,im.shape[1],2)
        y = np.arange(0,im.shape[0],2)
        interp_green2 = RegularGridInterpolator((y, x), im[1::2,::2], bounds_error=False, fill_value=None)
        im_green2 = interp_green2((Y,X))

        im_green = (im_green1 + im_green2) / 2

        # Interpolate blue channel
        x = np.arange(0,im.shape[1],2)
        y = np.arange(0,im.shape[0],2)
        interp_blue = RegularGridInterpolator((y, x), im[::2,::2], bounds_error=False, fill_value=None)
        im_blue = interp_blue((Y,X))

    else:
        raise NotImplementedError
    
    im_rgb = np.stack([im_red, im_green, im_blue], axis=-1)

    return im_rgb

def linear_brightening(im, target_brightness):
    """
    This implements the brightening step before gamma encoding.
    """

    im = im * target_brightness / np.mean(rgb2gray(im))
    im = np.clip(im, 0, 1)

    return im

def gamma_encoding(im):
    """
    This implements the gamma-encoding step.
    """

    thresh_mask = (im <= 0.0031308)

    im[thresh_mask] = 12.92 * im[thresh_mask]

    im[np.logical_not(thresh_mask)] = 1.055 * im[np.logical_not(thresh_mask)] ** (1/2.4) - 0.055

    return im

def wb_manual(im, pt_y, pt_x):

    r_val = im[pt_y, pt_x]
    g_val = (im[pt_y+1, pt_x] + im[pt_y, pt_x+1]) / 2
    b_val = im[pt_y+1, pt_x+1]

    im[::2,::2] = im[::2,::2] / r_val
    im[::2,1::2] = im[::2,1::2] / g_val
    im[1::2,::2] = im[1::2,::2] / g_val
    im[1::2,1::2] = im[1::2,1::2] / b_val

    im = im / np.max(im)

    return im