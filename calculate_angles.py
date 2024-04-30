

import cv2
import numpy as np
from skimage import io, color, morphology, filters, transform
from skimage.filters import gaussian
from skimage.morphology import skeletonize, binary_dilation

def calculate_angle(pt1, pt2, pt3):
    """计算由三个点pt1, pt2, pt3定义的角度"""
    vec1 = np.array([pt1[0] - pt2[0], pt1[1] - pt2[1]])
    vec2 = np.array([pt3[0] - pt2[0], pt3[1] - pt2[1]])
    cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    theta = np.arccos(max(min(cos_theta, 1.0), -1.0))  # 确保值在正确的范围内
    angle = np.degrees(theta)
    return angle

def calculate_distance(pt1, pt2):
    """计算两点之间的欧氏距离"""
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def calculate_angles(image_path, min_angle=90, min_distance=10):
    image = io.imread(image_path)
    gray_image = color.rgb2gray(image)
    blurred_image = gaussian(gray_image, sigma=1)
    thresh = filters.threshold_otsu(blurred_image)
    binary_image = blurred_image > thresh
    skeleton = skeletonize(binary_image)
    cleaned_skeleton = binary_dilation(skeleton)
    cleaned_skeleton_uint8 = (cleaned_skeleton * 255).astype(np.uint8)
    contours, _ = cv2.findContours(cleaned_skeleton_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marked_image = cv2.cvtColor(cleaned_skeleton_uint8, cv2.COLOR_GRAY2BGR)
    points_list = []

    for contour in contours:
        if len(contour) >= 3:
            for i in range(len(contour)):
                pt1 = tuple(contour[i % len(contour)][0])
                pt2 = tuple(contour[(i + 1) % len(contour)][0])
                pt3 = tuple(contour[(i + 2) % len(contour)][0])
                if calculate_angle(pt1, pt2, pt3) < min_angle:
                    if all(calculate_distance(pt2, existing_pt) > min_distance for existing_pt in points_list):
                        points_list.append(pt2)
                        cv2.circle(marked_image, pt2, 3, (0, 0, 255), -1)

    output_path = 'marked_straight_skeleton.png'
    cv2.imwrite(output_path, marked_image)
    cv2.imshow('Marked Image', marked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return len(points_list)

# Example usage:
num_angles = calculate_angles('plant_1892_666.jpg')
print(f"Number of distinct sharp angles: {num_angles}")
