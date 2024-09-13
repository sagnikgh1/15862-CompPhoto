import numpy as np

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
    tgt_idx = 2*np.argmax(im_brightness)
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

    else:
        raise NotImplementedError
    
    return im
        