import cv2
import numpy as np
import os
import csv
from skimage import io, color, morphology, filters, transform
from skimage.filters import gaussian

def calculate_pixels_per_cm(image_path='ruler.jpg', template_paths=['template_12.jpg', 'template_13.jpg']):
    ruler_image = cv2.imread(image_path, 0)
    templates = [cv2.imread(path, 0) for path in template_paths]
    scale_factors = np.linspace(0.5, 1.5, 20)
    best_scale = 0
    max_corr = 0
    best_locs = []

    for scale in scale_factors:
        scaled_templates = [cv2.resize(t, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for t in templates]
        for templ in scaled_templates:
            res = cv2.matchTemplate(ruler_image, templ, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > max_corr:
                max_corr = max_val
                best_scale = scale
                best_locs = [max_loc]

    if len(best_locs) >= 2:
        distance_in_pixels = abs(best_locs[0][1] - best_locs[1][1])
        actual_distance_cm = 1  # assuming '12' and '13' are 1 cm apart
        pixels_per_cm = distance_in_pixels / actual_distance_cm
        return pixels_per_cm
    return None

def calculate_length(image, pixels_per_cm):
    height, width, _ = image.shape
    length_cm = max(height, width) / pixels_per_cm
    return length_cm

def calculate_angle(pt1, pt2, pt3):
    """计算由三个点pt1, pt2, pt3定义的角度"""
    vec1 = np.array([pt1[0] - pt2[0], pt1[1] - pt2[1]])
    vec2 = np.array([pt3[0] - pt2[0], pt3[1] - pt2[1]])
    cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    theta = np.arccos(max(min(cos_theta, 1.0), -1.0))  # 确保值在正确的范围内
    angle = np.degrees(theta)
    return angle

def calculate_branches(image):
    gray_image = color.rgb2gray(image)
    blurred_image = gaussian(gray_image, sigma=1)
    thresh = filters.threshold_otsu(blurred_image)
    binary_image = blurred_image > thresh
    skeleton = morphology.skeletonize(binary_image)
    cleaned_skeleton = morphology.binary_dilation(skeleton)

    cleaned_skeleton_uint8 = (cleaned_skeleton * 255).astype(np.uint8)
    contours, _ = cv2.findContours(cleaned_skeleton_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    long_contours = [contour for contour in contours if cv2.arcLength(contour, True) > 40]

    branch_count = 0
    for contour in long_contours:
        if len(contour) >= 3:
            for i in range(len(contour)):
                pt1 = contour[i % len(contour)][0]
                pt2 = contour[(i + 1) % len(contour)][0]
                pt3 = contour[(i + 2) % len(contour)][0]
                angle = calculate_angle(pt1, pt2, pt3)
                if angle < 90:
                    branch_count += 1
    return branch_count

def process_images(directory, output_csv):
    pixels_per_cm = calculate_pixels_per_cm()
    if not pixels_per_cm:
        print("Failed to calculate pixels per cm.")
        return

    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'Length (cm)', 'Branches'])

        for filename in os.listdir(directory):
            if filename.startswith("plant") and filename.endswith(".jpg"):
                filepath = os.path.join(directory, filename)
                image = io.imread(filepath)

                length_cm = calculate_length(image, pixels_per_cm)
                branch_count = calculate_branches(image)

                writer.writerow([filename, length_cm, branch_count])
                os.remove(filepath)  # Optional: Remove file after processing

process_images('img', 'output_lengths_and_branches.csv')
