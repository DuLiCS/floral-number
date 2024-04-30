from skimage.io import imread
from skimage.morphology import skeletonize
from scipy.ndimage import convolve
import numpy as np

# Load the skeleton image
image_path = 'thick_skeleton.png'
skeleton_image = imread(image_path, as_gray=True)
skeleton = skeleton_image > 127  # Adjust this threshold based on your image

# Invert the skeleton if necessary
skeleton = np.invert(skeleton)

# Make sure it's a skeleton
skeleton = skeletonize(skeleton)

# Define a kernel to count neighbors
kernel = np.array([[1, 1, 1],
                   [1, 10, 1],
                   [1, 1, 1]], dtype=np.uint8)

# Convolve skeleton with the kernel
neighbor_count = convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)

# Endpoints have exactly one neighbor
endpoints = (neighbor_count == 11)

# Count endpoints
num_endpoints = np.sum(endpoints)
print(f"Number of endpoints: {num_endpoints}")
