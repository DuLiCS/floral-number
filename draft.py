from skimage import io, color, morphology, filters
from skimage.morphology import skeletonize, binary_opening, binary_closing, disk
from skimage.util import invert
import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_angle(pt1, pt2, pt3):
    vec1 = np.array(pt1) - np.array(pt2)
    vec2 = np.array(pt3) - np.array(pt2)
    unit_vec1 = vec1 / np.linalg.norm(vec1)
    unit_vec2 = vec2 / np.linalg.norm(vec2)
    dot_product = np.dot(unit_vec1, unit_vec2)
    angle = np.arccos(dot_product)
    return np.degrees(angle)


def find_branch_points(skeleton):
    branch_points = []
    # Use hit-or-miss transform or equivalent method to find junctions
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    hit_or_miss = morphology.binary_hit_or_miss(skeleton, selem=kernel)
    points = np.argwhere(hit_or_miss)

    # Distance threshold to consolidate nearby points
    min_distance = 15
    for p in points:
        if not any(np.linalg.norm(p - bp) < min_distance for bp in branch_points):
            branch_points.append(p)

    return branch_points


def process_image_for_branches(image_path):
    image = io.imread(image_path)
    gray_image = color.rgb2gray(image)
    blurred = filters.median(gray_image, disk(5))

    thresh = filters.threshold_otsu(blurred)
    binary = blurred > thresh
    binary = binary_closing(binary_opening(binary, disk(2)), disk(2))

    skeleton = skeletonize(binary)
    branch_points = find_branch_points(skeleton)
    return len(branch_points), skeleton, branch_points


# Use the function
branch_count, skeleton, branch_points = process_image_for_branches('plant_1892_666.jpg')
print(f"Detected branches: {branch_count}")

# Visualization
plt.imshow(skeleton, cmap='gray')
plt.scatter([p[1] for p in branch_points], [p[0] for p in branch_points], color='red', s=10)
plt.show()
