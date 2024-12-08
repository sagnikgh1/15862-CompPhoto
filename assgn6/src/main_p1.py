import os
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy

from cp_hw2 import lRGB2XYZ
from utils import shadowEdgeEstimation, shadowTimeEstimation, calibShadowLine, calibShadowPlane, reconstructPointCloud

def parse_args():

    parser = argparse.ArgumentParser(description='15862 Assignment 6')

    parser.add_argument('--data_dir', default='../data/frog', help='Path to data')

    parser.add_argument('--intrinsic_path', default='../data/calib/intrinsic_calib.npz', help='Path to intrinsic calib data')

    parser.add_argument('--extrinsic_path', default='../data/frog/extrinsic_calib.npz', help='Path to extrinsic calib data')

    args = parser.parse_args()

    return args


if __name__=="__main__":

    args = parse_args()

    # Read all images
    impath_ls = [f'{n:06}.jpg' for n in range(1,167)]
    I = [cv2.imread(os.path.join(args.data_dir, impath)) / 255 for impath in impath_ls]
    I_color = deepcopy(I[0])
    I_color = np.flip(I_color, axis=-1)
    I = [lRGB2XYZ(im)[:,:,1] for im in I]
    I = np.array(I)
    I = I.transpose((1,2,0))

    # Shadow edge estimation
    bbox_h = (200, 670, 820, 768)
    bbox_v = (250, 50, 820, 300)
    start_t, end_t = 57, 130
    line_coeffs_h, line_coeffs_v = shadowEdgeEstimation(I, bbox_h, bbox_v, start_t, end_t)

    # Visualize shadow edges
    t = 100
    coeffs_h = line_coeffs_h[t-start_t]
    coeffs_v = line_coeffs_v[t-start_t]
    ptx1 = - (coeffs_h[1]*bbox_h[1] + coeffs_h[2]) / coeffs_h[0]
    pt1 = (int(ptx1), bbox_h[1])
    ptx2 = - (coeffs_h[1]*bbox_h[3] + coeffs_h[2]) / coeffs_h[0]
    pt2 = (int(ptx2), bbox_h[3])
    ptx3 = - (coeffs_v[1]*bbox_v[1] + coeffs_v[2]) / coeffs_v[0]
    pt3 = (int(ptx3), bbox_v[1])
    ptx4 = - (coeffs_v[1]*bbox_v[3] + coeffs_v[2]) / coeffs_v[0]
    pt4 = (int(ptx4), bbox_v[3])
    im_viz = cv2.line(deepcopy(I[:,:,t]), pt1, pt2, (0,255,0), 9)
    im_viz = cv2.line(im_viz, pt3, pt4, (0,255,0), 9)
    plt.imsave("../data/data/shadowEdgeFrog2.jpg", im_viz, cmap='gray')

    # Shadow time estimation
    t_s = shadowTimeEstimation(I)
    plt.imsave("../data/data/shadowTime.jpg", t_s, cmap='jet')

    # Read intrinsic and extrinsic calibration matrices
    intrinsic_data = np.load(args.intrinsic_path)
    extrinsic_data = np.load(args.extrinsic_path)
    K = intrinsic_data['mtx']
    distortion = intrinsic_data['dist']
    R_h, t_h = extrinsic_data['rmat_h'], extrinsic_data['tvec_h']
    R_v, t_v = extrinsic_data['rmat_v'], extrinsic_data['tvec_v']

    # Shadow line calibration
    line_pts = calibShadowLine(line_coeffs_h, line_coeffs_v, bbox_h, bbox_v, K, distortion, R_h, t_h, R_v, t_v)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(line_pts[:,0,0], line_pts[:,0,1], line_pts[:,0,2])
    ax.scatter(line_pts[:,1,0], line_pts[:,1,1], line_pts[:,1,2])
    ax.scatter(line_pts[:,2,0], line_pts[:,2,1], line_pts[:,2,2])
    ax.scatter(line_pts[:,3,0], line_pts[:,3,1], line_pts[:,3,2])
    plt.show()
    np.savez("../data/data/line_pts.npz", line_pts)

    # Shadow plane calibration
    plane_pts, normal_vecs = calibShadowPlane(line_pts)
    np.savez("../data/data/shadow_planes.npz", line_pts)

    # Reconstruction
    bbox = (310, 310, 810, 650)
    plt.imshow(I[bbox[1]:bbox[3], bbox[0]:bbox[2], 120], cmap='gray')
    plt.show()
    pt_cloud, pt_cloud_color = reconstructPointCloud(I_color, t_s, bbox, plane_pts, normal_vecs, start_t, K, distortion)
    pt_cloud = np.array(pt_cloud)
    pt_cloud_color = np.array(pt_cloud_color)
    thresh = 5000
    median = np.median(pt_cloud, axis=1)
    thresh_mask = np.linalg.norm(pt_cloud-median[:,None], axis=1)<thresh
    pt_cloud = pt_cloud[thresh_mask]
    pt_cloud_color = pt_cloud_color[thresh_mask]
    xs = [pt_cloud[i][0] for i in range(len(pt_cloud))]
    ys = [pt_cloud[i][1] for i in range(len(pt_cloud))]
    zs = [pt_cloud[i][2] for i in range(len(pt_cloud))]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs, ys, zs, c=pt_cloud_color, s=1)
    plt.show()