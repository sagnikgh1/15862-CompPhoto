# Code Explanation
## main.py
This implements all parts for the provided data (door stack). Change the argparse arguments and look at the help messages for descriptions.
## main_captured.py
This implements all parts for the captured data (christmas stack). Change the argparse arguments and look at the help messages for descriptions.
## noise_calib.py
This estimates gain and additive noise by fitting a line through the mean-variance points. Change the last index in lines 46 and 47 to change the color channel.
## get_colorch_coords.py
This gets the coordinates of the colorchecker interactively. Click on top-left and bottom-right corners of the square you want to record. Go in row-major order of the colorcheckerboard for its default orientation.
## main_captured_optimal.py
Implements merging with optimal weights for captured data.
## utils.py
Helper functions.
