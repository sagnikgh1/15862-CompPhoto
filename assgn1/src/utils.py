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
    # Demosaic, without interpolation
    im_rgb = np.ones((im.shape[0], im.shape[1], 3))*np.nan

    if pattern=='grbg':
        im_rgb[1::2,::2,0] = im[1::2,::2,0] # Red
        