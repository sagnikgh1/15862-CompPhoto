import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse

from utils import RefocusUnstructuredLightField

def parse_args():

    parser = argparse.ArgumentParser(description='15862 Assignment 4')

    parser.add_argument('--ulf_path', default='../data/ulf', help='Path to unstructured lightfield images')

    args = parser.parse_args()

    return args

if __name__=="__main__":

    # Parse args
    args = parse_args()

    # Load unstructured lightfield
    frame_paths = os.listdir(args.ulf_path)
    frames = [cv2.imread(os.path.join(args.ulf_path, frame_paths[i])) for i in range(len(frame_paths))]
    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    frames = np.array(frames)

    # Refocus lightfield
    template_bbox = [170, 475, 205, 510]
    ref_frame_idx = 50
    search_bbox = [130, 435, 245, 550]
    refocused_im = RefocusUnstructuredLightField(frames, template_bbox, ref_frame_idx, search_bbox)
    plt.imsave("../data/refocused.jpg", refocused_im)