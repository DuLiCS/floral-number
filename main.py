import cv2
import numpy as np
from skimage import io, color, morphology, filters
from skimage.filters import gaussian
from matplotlib import pyplot as plt

def load_image(image_path, color_mode=cv2.IMREAD_GRAYSCALE):
    """加载图像并检查是否成功。"""
    img = cv2.imread(image_path, color_mode)
    if img is None:
        print("Image not found.")
        return None
    return img

def apply_template_matching(img, template):
    """应用模板匹配，并返回最大位置。"""
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    return max_loc

def rotate_image(image, angle):
    """旋转图像指定角度。"""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def visualize_and_save(img, points, filename='output.png'):
    """在图像上标记点并显示和保存图像。"""
    img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for point in points:
        top_left, bottom_right = point
        cv2.rectangle(img_display, top_left, bottom_right, (255, 0, 0), 2)
    cv2.imshow('Result', img_display)
    cv2.imwrite(filename, img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_image_for_analysis(image_path):
    """处理图像以进行进一步分析（如骨架化）。"""
    image = io.imread(image_path)
    gray_image = color.rgb2gray(image)
    blurred_image = gaussian(gray_image, sigma=1)
    thresh = filters.threshold_otsu(blurred_image)
    binary_image = blurred_image > thresh
    skeleton = morphology.skeletonize(binary_image)
    return skeleton

def find_and_draw_contours(binary_img, min_contour_length=40):
    """找到并绘制轮廓。"""
    contours, _ = cv2.findContours((binary_img * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    long_contours = [contour for contour in contours if cv2.arcLength(contour, True) > min_contour_length]
    return long_contours

# 示例用法:
image_path = 'ruler.jpg'  # Ruler image path
template_path = 'circle.jpg'  # Circle template path
img = load_image(image_path)
template = load_image(template_path)
max_loc = apply_template_matching(img, template)

# Determine if rotation is needed
w, h = template.shape[::-1]
vertical_center = max_loc[1] + h // 2
if vertical_center < img.shape[0] / 2:
    img = rotate_image(img, 180)
    print("Image rotated.")

points = [(max_loc, (max_loc[0] + w, max_loc[1] + h))]
visualize_and_save(img, points)

skeleton = process_image_for_analysis(image_path)
contours = find_and_draw_contours(skeleton)
# You can now process contours further as needed
