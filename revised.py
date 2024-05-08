import numpy as np
from matplotlib import pyplot as plt
from skimage import color, morphology, filters, util, transform
from skimage.filters import gaussian
import csv
import os
import cv2
from skimage import io
from scipy.ndimage import convolve
# 是否显示处理过程中的图像
show_flag = 0

# 函数：合并重叠的矩形，用于优化图像中的对象检测结果
def merge_overlapping_rectangles(rectangles):
    changed = True
    while changed:
        changed = False
        for i in range(len(rectangles)):
            for j in range(i + 1, len(rectangles)):
                if rectangles[i] and rectangles[j]:
                    # 获取两个矩形的坐标和大小
                    x1, y1, w1, h1 = rectangles[i]
                    x2, y2, w2, h2 = rectangles[j]
                    # 检查矩形是否重叠，并进行合并
                    if (x1 <= x2 + w2 and x1 + w1 >= x2 and y1 <= y2 + h2 and y1 + h1 >= y2):
                        min_x = min(x1, x2)
                        min_y = min(y1, y2)
                        max_x = max(x1 + w1, x2 + w2)
                        max_y = max(y1 + h1, y2 + h2)
                        new_rect = (min_x, min_y, max_x - min_x, max_y - min_y)
                        rectangles[i] = new_rect
                        rectangles[j] = None
                        changed = True
        rectangles = [r for r in rectangles if r]  # 移除已合并的矩形
    return rectangles

# 函数：根据指定角度旋转图像
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

# 函数：匹配模板并在必要时旋转图像
def match_template_and_rotate_if_necessary(image_path, template_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if img is None or template is None:
        print("未找到图像或模板。")
        return None

    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)

    w, h = template.shape[::-1]  # 模板大小
    vertical_center = max_loc[1] + h // 2  # 模板垂直中心
    img_height = img.shape[0]
    rotation_needed = 1 if vertical_center < img_height / 2 else 0

    if rotation_needed:
        img = rotate_image(img, 180)
        cv2.imwrite('ruler.jpg', img)
        print("图像已旋转并保存为 'ruler.jpg'")
    else:
        print("无需旋转。")

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img_display, top_left, bottom_right, (255, 0, 0), 2)
    if show_flag:
        cv2.imshow('模板匹配', img_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return rotation_needed


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

# 函数：计算植物的长度
def calculate_plant_length(rect, pixels_per_cm):
    x, y, w, h = rect
    plant_length_cm = max(w, h) / pixels_per_cm
    return plant_length_cm

# 函数：图像分割，用于提取图像中的重要部分
def image_segmentation(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, binary_thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(binary_thresh, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 计算矩形
    rectangles = [cv2.boundingRect(cnt) for cnt in contours if cv2.arcLength(cnt, True) > 550]
    merged_rectangles = merge_overlapping_rectangles(rectangles)

    # 去除不符合条件的矩形
    min_area = 60000  # 最小面积
    for rect in merged_rectangles:
        x, y, w, h = rect
        area = w * h
        aspect_ratio = w / float(h)
        if area > min_area:
            extracted = image[y:y + h, x:x + w]
            if aspect_ratio > 5 or aspect_ratio < 0.2:  # Assume ruler has a high or very low aspect ratio
                cv2.imwrite('ruler.jpg', extracted)
            else:
                cv2.imwrite('plant_{}_{}.jpg'.format(x, y), extracted)  # Naming by position to avoid overwrites

    image_with_rectangles = image.copy()
    for x, y, w, h in merged_rectangles:
        if w * h > min_area:
            cv2.rectangle(image_with_rectangles, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if show_flag:
        plt.imshow(cv2.cvtColor(image_with_rectangles, cv2.COLOR_BGR2RGB))
        plt.title('Processed Image with Extracted Objects')
        plt.axis('off')
        plt.show()

# 计算每厘米代表的像素数
def calculate_pixels_per_cm():
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
    if show_flag:
        # 显示图像
        cv2.imshow('Matched Templates', annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"Each cm represents {pixels_per_cm} pixel at scale factor {best_scale:.2f}")

    return pixels_per_cm


def calculate_angle(pt1, pt2, pt3):
    """计算由三个点pt1, pt2, pt3定义的角度"""
    vec1 = (pt1[0] - pt2[0], pt1[1] - pt2[1])
    vec2 = (pt3[0] - pt2[0], pt3[1] - pt2[1])
    inner_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    norm_vec1 = np.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
    norm_vec2 = np.sqrt(vec2[0] ** 2 + vec2[1] ** 2)
    cos_theta = inner_product / (norm_vec1 * norm_vec2)
    theta = np.arccos(cos_theta)
    angle = np.degrees(theta)
    return angle

# 计算植物长度的函数
def calculate_length_of_plants(directory, pixels_per_cm):
    lengths = []  # 用于存储所有图片的长度
    branch_nums = []  # 用于存储所有图片的分支数

    # 遍历当前目录下的所有文件
    for filename in os.listdir(directory):
        if filename.startswith("plant") and filename.endswith(".jpg"):
            # 构建完整的文件路径
            filepath = os.path.join(directory, filename)

            # 读取图像
            image = cv2.imread(filepath)
            if image is None:
                print(f"Failed to load {filename}")
                continue

            # 获取图像的高度和宽度
            height, width, _ = image.shape

            # 计算长度和宽度的厘米数
            length_cm = round(max(height, width) / pixels_per_cm, 2)
            lengths.append(length_cm)

            branch_count = skeleton_points(filepath, output_directory)  # 添加计算分支的函数调用
            branch_nums.append(branch_count)

            # 输出结果
            print(f"{filename}: Length = {length_cm:.2f} cm, Branches = {branch_count}")
            # 删除文件
            os.remove(filepath)
            print(f"Deleted {filename}")

    return lengths, branch_nums


def skeleton_points(image_path, output_directory):
    # 加载图像
    image = io.imread(image_path)
    if image.ndim == 3 and image.shape[2] == 4:  # 如果是RGBA格式，去除alpha通道
        image = image[..., :3]

    # 调整图像大小并标准化
    resized_image = transform.resize(image, (512, 512), anti_aliasing=True)

    # 转换为灰度图
    gray_image = color.rgb2gray(resized_image)

    # 应用高斯模糊
    blurred_image = filters.gaussian(gray_image, sigma=3)

    # 使用大津阈值进行二值化
    thresh = filters.threshold_otsu(blurred_image)
    binary_image = blurred_image > thresh

    # 骨架化
    skeleton = morphology.skeletonize(binary_image)

    # 检测端点
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)
    endpoints = (neighbor_count == 11)

    # 排除图像下四分之一和中央带区域的端点
    height, width = endpoints.shape
    bottom_quarter_start = 3 * height // 4
    central_band_start = height // 4
    central_band_end = 3 * height // 4

    endpoints[bottom_quarter_start:, :] = False
    endpoints[central_band_start:central_band_end, width // 4:3 * width // 4] = False

    # 将灰度图转换回RGB以便进行标记
    marked_image = color.gray2rgb(gray_image)

    # 通过标记更大的区域使端点更明显
    for y, x in np.argwhere(endpoints):
        size = 3  # 标记区域的半径
        slice_y = slice(max(0, y - size), min(marked_image.shape[0], y + size + 1))
        slice_x = slice(max(0, x - size), min(marked_image.shape[1], x + size + 1))

        marked_image[slice_y, slice_x, 0] = 1  # 红色通道
        marked_image[slice_y, slice_x, 1] = 0  # 绿色通道
        marked_image[slice_y, slice_x, 2] = 0  # 蓝色通道

    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)

    # 定义带有正确扩展名的保存路径
    save_path = os.path.join(output_directory, os.path.basename(image_path).split('.')[0] + '_marked.png')

    # 保存显示的图像以确保显示与保存一致
    plt.imsave(save_path, marked_image)
    print(f"已保存标记图像到 {save_path}")

    return np.sum(endpoints)

# 图片路径
img_directory = 'img'
# 模版路径
ruler_image_path = 'ruler.jpg'
template_path = 'circle.jpg'
output_directory = 'output'
# 存储结果的list
all_lengths = []
all_branch_nums = []

# 对文件夹内所有jpg文件进行操作
for filename in os.listdir(img_directory):
    if filename.endswith('.jpg'):
        # 图片绝对路径构建
        image_path = os.path.join(img_directory, filename)

        # 图片分割
        image_segmentation(image_path)

        rotation_flag = match_template_and_rotate_if_necessary(ruler_image_path, template_path)
        print(f"Rotation flag for {filename}: {rotation_flag} (1 means rotated, 0 means not rotated)")
        pixels_per_cm = calculate_pixels_per_cm()
        current_directory = os.getcwd()  # 获取当前工作目录
        length, branch_num = calculate_length_of_plants(current_directory, pixels_per_cm)
        print(f"{filename}: Length = {length}, Number of Branches = {branch_num}")

        # 保存结果
        all_lengths.append(length)
        all_branch_nums.append(branch_num)
# 结果写入csv
csv_file_path = 'plant_measurements.csv'
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image Name', 'Length (cm)', 'Number of Branches'])
    for i, filename in enumerate(os.listdir(img_directory)):
        if filename.endswith('.jpg'):
            writer.writerow([filename, all_lengths[i], all_branch_nums[i]])

print(f"Results saved to {csv_file_path}")
