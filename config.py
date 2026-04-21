import numpy as np

WORKING_DIR = "."

block_height = -45

default_port="COM8"


# TODO: This section needs to be updated with your calibration
#  3x3 affine matrix for pixel -> robot (X,Y)
M = np.array([
    [3.81562905e-04, -4.82264725e-01,  4.13297105e+02],
    [-4.58721299e-01,  3.58521982e-04,  1.36198619e+02]
], dtype=np.float64)

M[0, 2] -= 35.0  # shift +/-X
M[1, 2] += 8.0  # shift +/- y

z_above = 100           # safe travel height (e.g. 100)
z_table = -45           # Z at table contact
block_height_mm = 40   # block physical thickness
block_length_mm = 20   # block physical length
stack_delta_mm = 10    # extra height when stacking (to avoid collision)
side_offset_mm = 10    # extra XY gap when placing beside

capture_wait_time = 10
camera_index = 1