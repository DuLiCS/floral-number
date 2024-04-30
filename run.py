from skimage import io, img_as_bool, color, morphology, filters, transform
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import numpy as np
import cv2


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
    import cv2
    import numpy as np
    # 读取图像
    image = io.imread(image_path)

    # 调整图像大小
    #resized_image = transform.resize(image, (512, 512))  # 调整为 512x512 大小，可以根据实际情况调整大小

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
    epsilon_factor = 0.007

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

    # 显示标记过的图像
    cv2.imshow('Marked Image', marked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存标记过的图像
    output_path = 'marked_straight_skeleton.png'
    cv2.imwrite(output_path, marked_image)
    return sharp_angle_count - 1


a = calculate_angles('plant_1584_300.jpg')
print(a)