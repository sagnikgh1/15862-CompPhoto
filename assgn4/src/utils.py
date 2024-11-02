import os
import numpy as np
import cv2
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
from scipy import signal

from cp_hw2 import lRGB2XYZ

def LightfieldTo5dArr(im_lf, lenslet_size=16):
    """
    Converts the lightfield data to a 5D array (u,v,s,t,c).
    """
    h, w, _ = im_lf.shape

    im_5d = im_lf.reshape((h//lenslet_size, lenslet_size, w//lenslet_size, lenslet_size, 3))
    im_5d = im_5d.transpose((1,3,0,2,4))

    return im_5d

def CreateSubapertureMosaic(im_5d):
    """
    Creates a subaperture mosaic from the 5D lightfield array.
    """
    u, v, s, t, c = im_5d.shape

    im_5d = im_5d.transpose((0,2,1,3,4))
    im_mosaic = im_5d.reshape((u*s, v*t, c))

    return im_mosaic

def ShiftIm(im, shifts):
    """
    Shift given image by given shifts.
    """
    # Define interpolator
    y_coords = np.arange(im.shape[0])
    x_coords = np.arange(im.shape[1])
    interp = RegularGridInterpolator((y_coords, x_coords), im, bounds_error=False, fill_value=0)

    # Interpolate
    x_coords_eval = x_coords - shifts[1]
    y_coords_eval = y_coords - shifts[0]
    x_coords_eval, y_coords_eval = np.meshgrid(x_coords_eval, y_coords_eval)
    im_shifted = interp((y_coords_eval, x_coords_eval))

    return im_shifted

def RefocusLightfield(im_5d, d, aprtr_size=16):
    """
    Refocus the given lightfield to the given depth d.
    """
    u, v, s, t, c = im_5d.shape
    
    # Define UV grid
    maxUV = (u - 1) / 2
    U = np.arange(u) - maxUV
    V = np.arange(v) - maxUV

    # Crop aperture
    start_idx = int(U.shape[0]//2 - aprtr_size//2)
    end_ix = int(U.shape[0]//2 + aprtr_size//2)
    U = U[start_idx:end_ix]
    start_idx = int(V.shape[0]//2 - aprtr_size//2)
    end_ix = int(V.shape[0]//2 + aprtr_size//2)
    V = V[start_idx:end_ix]

    # Sum shifted subaperture images
    im_refocused = np.zeros((s, t, 3))
    for i in range(U.shape[0]):
        for j in range(V.shape[0]):
            shifts = (-d * U[i], d * V[j])
            subap_im = im_5d[i,j]
            subap_im_shifted = ShiftIm(subap_im, shifts)
            im_refocused += subap_im_shifted
    im_refocused = im_refocused / (U.shape[0] * V.shape[0])

    return im_refocused.astype(np.uint8)

def Linearize(im):
    """
    Approximately linearizes given image.
    """
    im_lin = np.zeros_like(im)
    im_lin[im<=0.0404482] = im[im<=0.0404482] / 12.92
    im_lin[im>0.0404482] = ((im[im>0.0404482] + 0.055) / 1.055) ** 2.4

    return im_lin

def GetLuminance(im):
    """
    Computes luminance of given image.
    """
    # Linearize
    im = Linearize(im / 255)

    # Convert to XYZ
    im = lRGB2XYZ(im)

    return im[:,:,1]

def GetLuminanceFS(im_fs):
    """
    Computes luminance of each image in a focal stack.
    """
    # Linearize focal stack
    im_fs = Linearize(im_fs / 255)

    # Convert to XYZ
    im_fs = np.stack([lRGB2XYZ(im_fs[:,:,:,i]) for i in range(im_fs.shape[-1])], axis=-1)

    return im_fs[:,:,1,:]

def DepthFromFocus(im_fs, depths, sigma1, sigma2):
    """
    Computes a depth map from a focal stack.
    """
    # Compute luminance
    im_fs_lum = GetLuminanceFS(im_fs)

    # Compute low frequency map
    im_fs_lf = cv2.GaussianBlur(im_fs_lum, ksize=(0,0), sigmaX=sigma1, sigmaY=sigma1)

    # Compute high frequency map
    im_fs_hf = im_fs_lum - im_fs_lf

    # Compute sharpness weights
    w_sharpness = cv2.GaussianBlur(im_fs_hf ** 2, ksize=(0,0), sigmaX=sigma2, sigmaY=sigma2)

    # Compute all-in-focus image
    im_aif = np.sum(w_sharpness[:,:,None,:] * im_fs, axis=-1) / np.sum(w_sharpness[:,:,None,:], axis=-1)

    # Compute depth map
    depth_map = np.sum(w_sharpness[:,:,:] * depths[None,None,::-1], axis=-1) / np.sum(w_sharpness[:,:,:], axis=-1)
    depth_map = depth_map / np.max(depth_map)
    depth_map = (depth_map * 255)

    return depth_map.astype(np.uint8), im_aif.astype(np.uint8)

def GenerateFocalApertureStack(im_5d, depths, apertures):
    """
    Create a focal-aperture stack from given lightfield.
    """
    u, v, s, t, c = im_5d.shape

    im_FAS = np.zeros((apertures.shape[0], depths.shape[0], s, t, c), dtype=im_5d.dtype)

    print(f"Generating FAS for {apertures.shape[0]} different aperture settings...")

    for i in tqdm(range(apertures.shape[0])):
        for j in range(depths.shape[0]):
            im_FAS[i,j] = RefocusLightfield(im_5d, depths[j], apertures[i])

    return im_FAS

def ConfocalStereo(im_fas, depths):
    """
    Obtain depth map from a focal-aperture stack using confocal stereo.
    """
    im_fas = np.mean(im_fas, axis=-1)

    # Compute variance along the aperture axis
    im_fas_var = np.var(im_fas, axis=0)

    # Compute depth map
    depth_map = np.argmin(im_fas_var, axis=0)
    depth_map = depth_map / np.max(depth_map)
    depth_map = (depth_map * 255)

    return depth_map.astype(np.uint8)

def Vid2Frames(vid_path, dst_path, step_frame, resize_fact):
    """
    Converts video to frames and saves to given path.
    """
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    cap = cv2.VideoCapture(vid_path)
    idx = 0
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = cv2.resize(frame, (0,0), fx=resize_fact, fy=resize_fact)
        if not idx%step_frame:
            fname = os.path.join(dst_path, f"{frame_idx:03}.jpg")
            cv2.imwrite(fname, frame)
            frame_idx += 1
        idx += 1

    cap.release()
    cv2.destroyAllWindows()


def ComputeShift(frame, template):
    """
    Computes the required shift of the given frame wrt to the ref frame.
    """
    # Compute luminance
    frame = GetLuminance(frame)
    template = GetLuminance(template)

    # Zero-mean
    template = template - np.mean(template)

    # Box filter frame
    kernel = np.ones_like(template)
    kernel = kernel / np.sum(kernel)
    frame_filt = cv2.filter2D(frame, -1, kernel)

    # Implement normalized crosscorrelation
    h_num = signal.correlate2d(frame, template, mode='same') # Numerator
    h_num -= frame_filt * np.sum(template)
    h_den = cv2.filter2D(frame ** 2, -1, kernel)
    h_den -= frame_filt ** 2
    h = h_num / np.sqrt(h_den)

    # Compute shift
    shift = np.argmax(h)
    shift = np.unravel_index(shift, h.shape)
    shift = np.array(shift)
    shift[0] = shift[0] - h.shape[0] / 2
    shift[1] = shift[1] - h.shape[1] / 2

    return -shift

def RefocusUnstructuredLightField(frames, template_bbox, ref_frame_idx, search_bbox):
    """
    Refocus the given frames of the unstructured light field.
    """
    template = frames[ref_frame_idx][template_bbox[1]:template_bbox[3],template_bbox[0]:template_bbox[2]]

    refocused_im = np.zeros((frames.shape[1], frames.shape[2], frames.shape[3]))

    for i in tqdm(range(frames.shape[0])):
        frame_search = frames[i][search_bbox[1]:search_bbox[3],search_bbox[0]:search_bbox[2]]
        shift = ComputeShift(frame_search, template)
        refocused_im += ShiftIm(frames[i], shift)

    refocused_im = refocused_im / frames.shape[0]

    return refocused_im.astype(np.uint8)