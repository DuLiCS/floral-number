import cv2
import numpy as np
from matplotlib import pyplot as plt

def merge_overlapping_rectangles(rectangles):
    changed = True
    while changed:
        changed = False
        for i in range(len(rectangles)):
            for j in range(i + 1, len(rectangles)):
                if rectangles[i] and rectangles[j]:
                    x1, y1, w1, h1 = rectangles[i]
                    x2, y2, w2, h2 = rectangles[j]
                    if (x1 <= x2 + w2 and x1 + w1 >= x2 and y1 <= y2 + h2 and y1 + h1 >= y2):
                        min_x = min(x1, x2)
                        min_y = min(y1, y2)
                        max_x = max(x1 + w1, x2 + w2)
                        max_y = max(y1 + h1, y2 + h2)
                        new_rect = (min_x, min_y, max_x - min_x, max_y - min_y)
                        rectangles[i] = new_rect
                        rectangles[j] = None
                        changed = True
        rectangles = [r for r in rectangles if r]
    return rectangles

def calculate_plant_length(rect, pixels_per_cm):
    x, y, w, h = rect
    plant_length_cm = max(w, h) / pixels_per_cm
    return plant_length_cm

# Load and preprocess the image
image_path = 'img/1.jpg'
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
_, binary_thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
edges = cv2.Canny(binary_thresh, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calculate bounding rectangles
rectangles = [cv2.boundingRect(cnt) for cnt in contours if cv2.arcLength(cnt, True) > 550]
merged_rectangles = merge_overlapping_rectangles(rectangles)

# Define the pixels per cm from a previously calculated function or input
pixels_per_cm = 121  # Example value, replace with actual calculated value

# Filter and calculate plant lengths
min_area = 50000
image_with_rectangles = image.copy()
for rect in merged_rectangles:
    x, y, w, h = rect
    area = w * h
    aspect_ratio = w / float(h)
    extracted = image[y:y + h, x:x + w]
    if area > min_area and not (aspect_ratio > 5 or aspect_ratio < 0.2):  # Exclude the ruler based on aspect ratio
        plant_length_cm = calculate_plant_length(rect, pixels_per_cm)
        print(f"Plant at ({x}, {y}) with size ({w}x{h} pixels) has length: {plant_length_cm:.2f} cm")
        cv2.rectangle(image_with_rectangles, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image_with_rectangles, f"{plant_length_cm:.2f} cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

# Display the processed image with rectangles
plt.imshow(cv2.cvtColor(image_with_rectangles, cv2.COLOR_BGR2RGB))
plt.title('Processed Image with Extracted Plants and Their Lengths')
plt.axis('off')
plt.show()




######################################
# 计算分支数量


