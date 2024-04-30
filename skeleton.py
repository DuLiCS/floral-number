import matplotlib.pyplot as plt
from skimage import io, color, morphology, filters, transform
from skimage.filters import gaussian

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
    return num_endpoints


print(skeleton_points('plant_1733_52.jpg'))