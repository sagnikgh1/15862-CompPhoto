import numpy as np
import cv2

from cp_hw6 import pixel2ray

def shadowEdgeEstimation(I, bbox_h, bbox_v, start_t, end_t):
    """
    Estimate shadow edges from a video I(x,y,t).
    """
    # Compute I_shadow
    I_max = np.max(I, axis=-1)
    I_min = np.min(I, axis=-1)
    I_shad = (I_max + I_min) / 2

    # Compute difference image
    I_diff = I - I_shad[:,:,None]

    # Horizontal shadow edge estimation
    I_diff_h = I_diff[bbox_h[1]:bbox_h[3], bbox_h[0]:bbox_h[2], start_t:end_t]
    line_coeffs_h = fitLine(I_diff_h, bbox_h[1], bbox_h[0])

    # Vertical shadow edge estimation
    I_diff_v = I_diff[bbox_v[1]:bbox_v[3], bbox_v[0]:bbox_v[2], start_t:end_t]
    line_coeffs_v = fitLine(I_diff_v, bbox_v[1], bbox_v[0])

    return line_coeffs_h, line_coeffs_v

def fitLine(I_diff, start_y, start_x):
    """
    Given the difference image for one time t, return params of shadow line.
    """
    line_coeffs = np.zeros((I_diff.shape[-1],3))
    for t in range(I_diff.shape[-1]):
        pt_ls = [] # Zero-crossing list
        for i in range(I_diff.shape[0]):
            for j in range(I_diff.shape[1]-1):
                if I_diff[i,j,t]<0 and I_diff[i,j+1,t]>0:
                    pt_ls.append([j+1+start_x, i+start_y])
                    break
        
        # Fit a line
        A = np.array(pt_ls)
        A = np.concatenate([A, np.ones((A.shape[0],1))], axis=1)
        U, S, Vh = np.linalg.svd(A)
        x = Vh[-1]
        line_coeffs[t] = x
    return line_coeffs

def shadowTimeEstimation(I, I_thresh=0.2):
    """
    Estimate shadow time frame from a video I(x,y,t).
    """
    # Compute I_shadow
    I_max = np.max(I, axis=-1)
    I_min = np.min(I, axis=-1)
    I_shad = (I_max + I_min) / 2

    # Compute difference image
    I_diff = I - I_shad[:,:,None]

    # For each pixel, find zero-crossing
    t_s = np.argmin(I_diff[:,:,:-1] - I_diff[:,:,1:], axis=-1)
    
    # Threshold t_s
    thresh_mask = ((I_max - I_min) < I_thresh)
    t_s[thresh_mask] = 0

    return t_s

def calibShadowLineFrame(coeffs_h, coeffs_v, bbox_h, bbox_v, K, distortion, R_h, t_h, R_v, t_v):
    """
    Estimate 3D coordinates of four points on the shadow line of each frame.
    """
    # Estimate image locations of the four points
    p1x = - (coeffs_h[1]*bbox_h[3] + coeffs_h[2]) / coeffs_h[0]
    p1 = np.array([p1x, bbox_h[3]])
    p2x = - (coeffs_h[1]*bbox_h[1] + coeffs_h[2]) / coeffs_h[0]
    p2 = np.array([p2x, bbox_h[1]])
    p3x = - (coeffs_v[1]*bbox_v[3] + coeffs_v[2]) / coeffs_v[0]
    p3 = np.array([p3x, bbox_v[3]])
    p4x = - (coeffs_v[1]*bbox_v[1] + coeffs_v[2]) / coeffs_v[0]
    p4 = np.array([p4x, bbox_v[1]])

    # Project rays from pixels
    r1 = pixel2ray(p1[None], K, distortion)[0].T
    r2 = pixel2ray(p2[None], K, distortion)[0].T
    r3 = pixel2ray(p3[None], K, distortion)[0].T
    r4 = pixel2ray(p4[None], K, distortion)[0].T

    # Convert to plane coords
    r1 = R_h.T @ r1
    r2 = R_h.T @ r2
    r3 = R_v.T @ r3
    r4 = R_v.T @ r4

    # Camera center in horizontal and vertical plane coords
    Pc_h = - R_h.T @ t_h
    Pc_v = - R_v.T @ t_v

    # Line points in plane coords
    P1 = Pc_h - (Pc_h[-1,0] / r1[-1,0]) * r1
    P2 = Pc_h - (Pc_h[-1,0] / r2[-1,0]) * r2
    P3 = Pc_v - (Pc_v[-1,0] / r3[-1,0]) * r3
    P4 = Pc_v - (Pc_v[-1,0] / r4[-1,0]) * r4

    # Convert line points back to camera coords
    P1 = R_h @ P1 + t_h
    P2 = R_h @ P2 + t_h
    P3 = R_v @ P3 + t_v
    P4 = R_v @ P4 + t_v

    return np.squeeze(np.array([P1, P2, P3, P4]))

def calibShadowLine(coeffs_h, coeffs_v, bbox_h, bbox_v, K, distortion, R_h, t_h, R_v, t_v):
    """
    Estimate 3D coordinates of four points on shadow line for all planes.
    """
    line_pts = [calibShadowLineFrame(coeffs_h[t], coeffs_v[t], bbox_h, bbox_v, K, distortion, R_h, t_h, R_v, t_v) for t in range(coeffs_h.shape[0])]
    line_pts = np.array(line_pts)

    return line_pts

def calibShadowPlane(line_pts):
    """
    Compute parameters of shadow plane of each frame.
    """
    normal_vecs = np.cross(line_pts[:,1] - line_pts[:,0], line_pts[:,3] - line_pts[:,2])
    normal_vecs = normal_vecs / np.linalg.norm(normal_vecs, axis=1, keepdims=True)

    return line_pts[:,0], normal_vecs

def reconstructPointCloud(I_color, t_s, bbox, plane_pts, normal_vecs, start_t, K, distortion):
    """
    Reconstruct point cloud.
    """
    # Crop using bbox
    t_s = t_s
    I_color = I_color

    # Compute point cloud
    pt_cloud = []
    pt_cloud_color = []
    for i in range(bbox[1], bbox[3]):
        for j in range(bbox[0], bbox[2]):
            t_s_curr = int(t_s[i,j] - start_t)
            # t_s_curr = max(0, min(t_s_curr, 72))
            # t_s_curr = max(0, min(t_s_curr, 69))
            t_s_curr = max(0, min(t_s_curr, 169))
            plane_pt = plane_pts[t_s_curr][:,None]
            normal_vec = normal_vecs[t_s_curr][:,None]
            ray = pixel2ray(np.array([j, i])[None].astype(np.float32),  K, distortion)[0].T
            intersection_pt = ray * np.sum(normal_vec * plane_pt) / np.sum(normal_vec * ray)
            pt_cloud.append(np.squeeze(intersection_pt))
            pt_cloud_color.append(I_color[i,j])
    
    return pt_cloud, pt_cloud_color