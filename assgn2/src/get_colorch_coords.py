import numpy as np
import matplotlib.pyplot as plt
import cv2

if __name__=="__main__":
    # Read image
    im = cv2.imread("../data/door_stack/exposure16.jpg")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # Get list of coords
    plt.imshow(im)
    coords_ls = plt.ginput(n=48, timeout=1000)
    coords_ls = np.array(coords_ls)
    np.save("../data/colorch_coords.npy", coords_ls)