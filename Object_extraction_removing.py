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

# Load and preprocess the image
image_path = 'img/4.jpg'
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
_, binary_thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
edges = cv2.Canny(binary_thresh, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calculate bounding rectangles
rectangles = [cv2.boundingRect(cnt) for cnt in contours if cv2.arcLength(cnt, True) > 550]
merged_rectangles = merge_overlapping_rectangles(rectangles)

# Filter and save the results
min_area = 50000  # Minimum area threshold
for rect in merged_rectangles:
    x, y, w, h = rect
    area = w * h
    aspect_ratio = w / float(h)
    if area > min_area:
        extracted = image[y:y+h, x:x+w]
        if aspect_ratio > 5 or aspect_ratio < 0.2:  # Assume ruler has a high or very low aspect ratio
            cv2.imwrite('ruler.jpg', extracted)
        else:
            cv2.imwrite('plant_{}_{}.jpg'.format(x, y), extracted)  # Naming by position to avoid overwrites

# Display the processed image with rectangles
image_with_rectangles = image.copy()
for x, y, w, h in merged_rectangles:
    if w * h > min_area:  # Directly using the rectangle dimensions
        cv2.rectangle(image_with_rectangles, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(image_with_rectangles, cv2.COLOR_BGR2RGB))
plt.title('Processed Image with Extracted Objects')
plt.axis('off')
plt.show()
