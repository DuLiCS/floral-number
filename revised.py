import numpy as np
from matplotlib import pyplot as plt
from skimage import color, morphology, filters
from skimage.filters import gaussian
import csv
import os
import cv2
from skimage import io

show_flag = 0


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
        print("Image rotated and saved as 'ruler.jpg'")
    else:
        print("No rotation needed.")

    # Draw rectangle for visualization
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img_display, top_left, bottom_right, (255, 0, 0), 2)
    if show_flag:
        cv2.imshow('Template Matching', img_display)
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


def calculate_plant_length(rect, pixels_per_cm):
    x, y, w, h = rect
    plant_length_cm = max(w, h) / pixels_per_cm
    return plant_length_cm


def image_segmentation(image_path):
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
    min_area = 60000  # Minimum area threshold
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

    # Display the processed image with rectangles
    image_with_rectangles = image.copy()
    for x, y, w, h in merged_rectangles:
        if w * h > min_area:  # Directly using the rectangle dimensions
            cv2.rectangle(image_with_rectangles, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if show_flag:
        plt.imshow(cv2.cvtColor(image_with_rectangles, cv2.COLOR_BGR2RGB))
        plt.title('Processed Image with Extracted Objects')
        plt.axis('off')
        plt.show()


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


def calculate_angles(image_path):
    # 读取图像
    image = io.imread(image_path)

    # 调整图像大小
    # resized_image = transform.resize(image, (512, 512))  # 调整为 512x512 大小，可以根据实际情况调整大小

    # 转换为灰度图像
    gray_image = color.rgb2gray(image)

    # 应用高斯模糊
    blurred_image = gaussian(gray_image, sigma=1)  # sigma 控制模糊的程度

    # 应用阈值来二值化图像
    # 使用大津算法找到一个阈值
    thresh = filters.threshold_otsu(blurred_image)
    binary_image = blurred_image > thresh

    # 应用骨架化算法
    skeleton = morphology.skeletonize(binary_image)

    # 试着使用更小的形态学操作，或者减少操作的强度
    from skimage.morphology import binary_dilation
    cleaned_skeleton = binary_dilation(skeleton)  # 使用轻微的膨胀可能有助于恢复某些细节

    if show_flag:
        # 显示原图、骨架图和去毛刺后的骨架图
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].imshow(binary_image, cmap=plt.cm.gray)
        ax[0].axis('off')
        ax[0].set_title('Binary Image', fontsize=20)

        ax[1].imshow(skeleton, cmap=plt.cm.gray)
        ax[1].axis('off')
        ax[1].set_title('Initial Skeleton', fontsize=20)

        ax[2].imshow(cleaned_skeleton, cmap=plt.cm.gray)
        ax[2].axis('off')
        ax[2].set_title('Cleaned Skeleton', fontsize=20)

        fig.tight_layout()
        plt.show()

    # Convert the cleaned skeleton to uint8 before finding contours
    cleaned_skeleton_uint8 = (cleaned_skeleton * 255).astype(np.uint8)

    # 设置轮廓的最小长度阈值
    min_contour_length = 40

    # 查找所有的轮廓
    contours, _ = cv2.findContours(cleaned_skeleton_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 过滤掉太短的轮廓
    long_contours = [contour for contour in contours if cv2.arcLength(contour, True) > min_contour_length]

    # 创建一个全黑的背景
    straight_lines_img = np.zeros_like(cleaned_skeleton_uint8)

    # 设置近似的精确度
    epsilon_factor = 0.008

    # 对于每个较长轮廓，应用多边形近似，并画出结果
    for contour in long_contours:
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(straight_lines_img, [approx], -1, (255), 1)

    # 输出图像
    output_path = 'straight_skeleton.png'
    cv2.imwrite(output_path, straight_lines_img)

    # 加载骨架图像
    image = cv2.imread('straight_skeleton.png', cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个彩色图像以标记尖角
    marked_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    # 对于每个轮廓
    sharp_angle_count = 0
    for contour in contours:
        # 需要至少3个点来形成角
        if len(contour) >= 3:
            for i in range(len(contour)):
                pt1 = contour[i % len(contour)][0]
                pt2 = contour[(i + 1) % len(contour)][0]
                pt3 = contour[(i + 2) % len(contour)][0]
                angle = calculate_angle(pt1, pt2, pt3)

                # 如果角度小于90度，则在图像上标记
                if angle < 90:
                    sharp_angle_count += 1
                    cv2.circle(marked_image, tuple(pt2), 3, (0, 0, 255), -1)  # 使用红色圆圈标记
    if show_flag:
        # 显示标记过的图像
        cv2.imshow('Marked Image', marked_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 保存标记过的图像
    output_path = 'marked_straight_skeleton.png'
    cv2.imwrite(output_path, marked_image)
    return sharp_angle_count - 1


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
            length_cm = max(height, width) / pixels_per_cm
            lengths.append(length_cm)

            # 此处假设calculate_angles是前文定义的计算分支数的函数
            branch_count = skeleton_points(filepath)  # 添加计算分支的函数调用
            branch_nums.append(branch_count)

            # 输出结果
            print(f"{filename}: Length = {length_cm:.2f} cm, Branches = {branch_count}")
            # 删除文件
            os.remove(filepath)
            print(f"Deleted {filename}")

    return lengths, branch_nums


def skeleton_points(image_path):
    from skimage import io, color, morphology, filters, transform

    # 读取图
    image = io.imread(image_path)

    # 调整图像大小
    resized_image = transform.resize(image, (512, 512))  # 调整为 512x512 大小，可以根据实际情况调整大小

    # 转换为灰度图像
    gray_image = color.rgb2gray(resized_image)

    # 应用高斯模糊
    blurred_image = gaussian(gray_image, sigma=1)  # sigma 控制模糊的程度

    # 应用阈值来二值化图像
    # 使用大津算法找到一个阈值
    thresh = filters.threshold_otsu(blurred_image)
    binary_image = blurred_image > thresh

    # 应用骨架化算法
    skeleton = morphology.skeletonize(binary_image)

    # 试着使用更小的形态学操作，或者减少操作的强度
    from skimage.morphology import binary_dilation
    cleaned_skeleton = binary_dilation(skeleton)  # 使用轻微的膨胀可能有助于恢复某些细节

    # 显示原图、骨架图和去毛刺后的骨架图
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), sharex=True, sharey=True)
    ax = axes.ravel()
    if show_flag:
        ax[0].imshow(binary_image, cmap=plt.cm.gray)
        ax[0].axis('off')
        ax[0].set_title('Binary Image', fontsize=20)

        ax[1].imshow(skeleton, cmap=plt.cm.gray)
        ax[1].axis('off')
        ax[1].set_title('Initial Skeleton', fontsize=20)

        ax[2].imshow(cleaned_skeleton, cmap=plt.cm.gray)
        ax[2].axis('off')
        ax[2].set_title('Cleaned Skeleton', fontsize=20)

        fig.tight_layout()
        plt.show()

    # 保存去毛刺后的骨架图像
    skeleton_image_path = 'cleaned_skeleton.png'
    plt.imsave(skeleton_image_path, cleaned_skeleton, cmap=plt.cm.gray)

    from skimage import io, morphology, util, color
    from skimage.morphology import skeletonize
    from skimage.util import img_as_bool

    # 加载骨架图像
    skeleton_image = io.imread('cleaned_skeleton.png')

    # 将图像转换为布尔值
    skeleton = util.img_as_bool(skeleton_image)

    # 如果图像有4个通道，假设最后一个通道是alpha通道，可以舍弃
    if skeleton.ndim == 3 and skeleton.shape[2] == 4:
        # 只取前三个通道
        skeleton = skeleton[..., :3]

    # 将图像转换为灰度
    skeleton_gray = color.rgb2gray(skeleton)

    # 确保图像是布尔类型
    skeleton_bool = img_as_bool(skeleton_gray)

    selem = morphology.disk(1)  # 可以调整大小

    # 现在执行形态学操作
    cleaned = morphology.binary_opening(skeleton_bool, selem)
    # 使用形态学开运算去除毛刺


    # 可选：移除小的连通组件
    cleaned = morphology.remove_small_objects(cleaned, min_size=100)  # min_size 也可以调整

    # 可选：继续使用其他形态学操作如闭运算来平滑结果
    # selem = morphology.disk(1)  # 可以调整大小
    cleaned = morphology.binary_closing(cleaned, selem)

    # 骨架化处理后的清理图像
    cleaned_skeleton = skeletonize(cleaned)

    # 保存处理后的图像
    io.imsave('cleaned_skeleton_without_spurs.png', util.img_as_ubyte(cleaned_skeleton))





    from skimage.io import imread, imsave
    from skimage.morphology import skeletonize, binary_dilation, disk
    from skimage.util import img_as_bool, invert
    from skimage import img_as_ubyte  # Import this function

    # Read the skeleton image
    skeleton_image_path = 'cleaned_skeleton_without_spurs.png'
    skeleton_image = imread(skeleton_image_path, as_gray=True)

    # Convert to boolean array
    skeleton_bool = img_as_bool(skeleton_image)

    # Dilate the skeleton image to make it thicker, this may remove small branches
    dilated_skeleton = binary_dilation(skeleton_bool, disk(50))

    # Skeletonize again to make sure it's a single-pixel skeleton
    thick_skeleton = skeletonize(dilated_skeleton)

    # Convert the image to uint8 format before saving
    thick_skeleton_uint8 = img_as_ubyte(invert(thick_skeleton))

    # Save the resulting image
    thick_skeleton_image_path = 'thick_skeleton.png'
    imsave(thick_skeleton_image_path, thick_skeleton_uint8)  # Save as uint8 image

    # Provide the path of the saved image
    print(thick_skeleton_image_path)


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
    return num_endpoints - 1


# Assuming the following functions are defined:
# - image_segmentation(image_path)
# - match_template_and_rotate_if_necessary(image_path, template_path)
# - calculate_pixels_per_cm()
# - calculate_length_of_plants(directory, pixels_per_cm)
# - calculate_angles(image_path)

# Directory containing the images
img_directory = 'img'

# Path to the ruler and template images
ruler_image_path = 'ruler.jpg'
template_path = 'circle.jpg'

# Calculate pixels per cm using the ruler image


# Prepare lists to store the results
all_lengths = []
all_branch_nums = []

# Iterate through all .jpg files in the img directory
for filename in os.listdir(img_directory):
    if filename.endswith('.jpg'):
        # Construct the full image path
        image_path = os.path.join(img_directory, filename)

        # Perform image segmentation
        image_segmentation(image_path)

        # Perform template matching and possible rotation
        rotation_flag = match_template_and_rotate_if_necessary(ruler_image_path, template_path)
        print(f"Rotation flag for {filename}: {rotation_flag} (1 means rotated, 0 means not rotated)")
        pixels_per_cm = calculate_pixels_per_cm()
        # Calculate the length and number of branches
        current_directory = os.getcwd()  # 获取当前工作目录

        length, branch_num = calculate_length_of_plants(current_directory, pixels_per_cm)
        print(f"{filename}: Length = {length}, Number of Branches = {branch_num}")

        # Store the results
        all_lengths.append(length)
        all_branch_nums.append(branch_num)
        1 == 1
# Write the results to a CSV file
csv_file_path = 'plant_measurements.csv'
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['Image Name', 'Length (cm)', 'Number of Branches'])
    # Write the data
    for i, filename in enumerate(os.listdir(img_directory)):
        if filename.endswith('.jpg'):
            writer.writerow([filename, all_lengths[i], all_branch_nums[i]])

print(f"Results saved to {csv_file_path}")
