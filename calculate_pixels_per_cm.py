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
image_path = 'img/30.jpg'
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



import cv2
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def match_template_and_rotate_if_necessary(image_path, template_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if img is None or template is None:
        print("Image or template not found.")
        return None

    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)

    # Determine the size of the template
    w, h = template.shape[::-1]

    # Calculate the vertical center of the found template
    vertical_center = max_loc[1] + h // 2
    img_height = img.shape[0]
    rotation_needed = 1 if vertical_center < img_height / 2 else 0

    # If the circle is in the upper half, rotate the image
    if rotation_needed:
        img = rotate_image(img, 180)
        # Save the rotated image
        cv2.imwrite('ruler.jpg', img)
        print("Image rotated and saved as 'rotated_ruler.jpg'")
    else:
        print("No rotation needed.")

    # Draw rectangle for visualization
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img_display, top_left, bottom_right, (255, 0, 0), 2)
    cv2.imshow('Template Matching', img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return rotation_needed

# Paths to the images
image_path = 'ruler.jpg'  # Ruler image path
template_path = 'circle.jpg'  # Circle template path
rotation_flag = match_template_and_rotate_if_necessary(image_path, template_path)
print(f"Rotation flag: {rotation_flag} (1 means rotated, 0 means not rotated)")


##############################


import cv2
import numpy as np

# 加载尺子的图像
image_path = 'ruler.jpg'
ruler_image = cv2.imread(image_path, 0)  # 以灰度模式加载

# 加载“12”和“13”的模板
template_paths = ['template_12.jpg', 'template_13.jpg']
templates = [cv2.imread(path, 0) for path in template_paths]

# 搜索窗口的尺寸，用于控制模板缩放的大小范围
scale_factors = np.linspace(0.5, 1.5, 20)  # 范围从0.5到1.5，总共20个尺度

best_scale = 0
max_corr = 0
locations = []
best_locs = []

# 在多个尺度上执行模板匹配
for scale in scale_factors:
    # 为每个模板应用缩放
    scaled_templates = [cv2.resize(t, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for t in templates]
    match_vals = []

    # 计算每个缩放后的模板的匹配值
    for templ in scaled_templates:
        res = cv2.matchTemplate(ruler_image, templ, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        match_vals.append((max_val, max_loc))

    # 选取当前尺度下匹配度最高的结果
    current_max_corr = max(match_vals, key=lambda x: x[0])[0]
    if current_max_corr > max_corr:
        max_corr = current_max_corr
        best_scale = scale
        best_locs = [val[1] for val in match_vals]  # 更新最佳匹配位置

# 使用最佳缩放尺度下的匹配位置
distance_in_pixels = abs(best_locs[0][1] - best_locs[1][1])

# 假设“12”和“13”之间的实际距离为1厘米
actual_distance_cm = 1
pixels_per_cm = distance_in_pixels / actual_distance_cm

# 在图像上绘制矩形标记匹配的位置
annotated_image = cv2.cvtColor(ruler_image, cv2.COLOR_GRAY2BGR)
for idx, loc in enumerate(best_locs):
    templ = templates[idx]
    cv2.rectangle(annotated_image, loc, (loc[0] + templ.shape[1], loc[1] + templ.shape[0]), (0, 255, 0), 2)

# 显示图像
cv2.imshow('Matched Templates', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Each cm represents {pixels_per_cm} pixel at scale factor {best_scale:.2f}")




####################

import cv2
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

# Filter and calculate plant lengths
min_area = 50000
image_with_rectangles = image.copy()
for rect in merged_rectangles:
    x, y, w, h = rect
    area = w * h
    aspect_ratio = w / float(h)
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



