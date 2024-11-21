import numpy as np
import cv2

def DecomposeI(I):
    """
    Decompose I matrix (7xP) into L and B.
    """
    # Compute SVD
    U, S, Vh = np.linalg.svd(I, full_matrices=False)

    # Truncate to rank 3
    U = U[:,:3]
    S = S[:3]
    Vh = Vh[:3,:]

    # Compute L and B
    L = (U @ np.sqrt(np.diag(S))).T
    B = np.sqrt(np.diag(S)) @ Vh

    return L, B

def EnforceIntegrability(B, shp, sigma):
    """
    Transform B such that integrability is enforced.
    """
    # Reshape B
    B_im = B.T
    B_im = B_im.reshape((shp[0], shp[1], 3))

    # Blur B and compute gradients
    B_blur = cv2.GaussianBlur(B_im, ksize=(0,0), sigmaX=sigma, sigmaY=sigma)
    B_gradY, B_gradX = np.gradient(B_blur, axis=(0,1))
    B_gradY = B_gradY.reshape((-1,3))
    B_gradY = B_gradY.T
    B_gradX = B_gradX.reshape((-1,3))
    B_gradX = B_gradX.T

    # Build least squares matrix
    A1 = B[0,:] * B_gradX[1,:] - B[1,:] * B_gradX[0,:]
    A2 = B[0,:] * B_gradX[2,:] - B[2,:] * B_gradX[0,:]
    A3 = B[1,:] * B_gradX[2,:] - B[2,:] * B_gradX[1,:]
    A4 = -B[0,:] * B_gradY[1,:] + B[1,:] * B_gradY[0,:]
    A5 = -B[0,:] * B_gradY[2,:] + B[2,:] * B_gradY[0,:]
    A6 = -B[1,:] * B_gradY[2,:] + B[2,:] * B_gradY[1,:]
    A = np.stack([A1,A2,A3,A4,A5,A6], axis=1)

    # Compute SVD
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    x = Vh[-1]
    Q = np.array([[-x[2], x[5], 1],
                  [x[1], -x[4], 0],
                  [-x[0], x[3], 0]])
    Q = Q.T

    # Apply transformation
    B = np.linalg.inv(Q).T @ B

    return B