import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize, binary_opening, disk, remove_small_objects
from skimage.feature import peak_local_max
from sklearn.cluster import KMeans
from scipy.ndimage import convolve

def process_image(image_path):
    # 加载和处理图像
    image = imread(image_path)
    if image.ndim == 3:
        image = rgb2gray(image)  # 转换为灰度
    thresh = threshold_otsu(image)
    binary = image > thresh
    skeleton = skeletonize(binary)

    # 清理骨架
    selem = disk(1)
    cleaned_skeleton = binary_opening(skeleton, selem)
    cleaned_skeleton = remove_small_objects(cleaned_skeleton, min_size=64)

    # 检测尖端
    endpoints = detect_endpoints(cleaned_skeleton)

    # 聚类尖端
    if endpoints.any():
        cluster_labels = cluster_endpoints(endpoints)
        plot_clusters(cleaned_skeleton, endpoints, cluster_labels)
    else:
        print("No endpoints detected.")

def detect_endpoints(skeleton):
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbors = convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)
    endpoints = (neighbors == 11)
    return endpoints

def cluster_endpoints(endpoints):
    points = np.column_stack(np.where(endpoints))
    if len(points) > 1:
        kmeans = KMeans(n_clusters=min(10, len(points)//2))
        kmeans.fit(points)
        labels = kmeans.labels_
    else:
        labels = np.array([0])
    return labels

def plot_clusters(skeleton, endpoints, labels):
    plt.imshow(skeleton, cmap='gray')
    points = np.column_stack(np.where(endpoints))
    scatter = plt.scatter(points[:, 1], points[:, 0], c=labels, cmap='rainbow', edgecolor='white')
    plt.title(f"Total Clusters: {len(np.unique(labels))}")
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.axis('off')
    plt.show()

# 调用函数处理图像
process_image('plant_1733_52.jpg')
