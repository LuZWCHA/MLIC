import csv
from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool, cpu_count
from PIL import Image

def load_image(image_path):
    """加载图像并转换为灰度图像"""
    img = Image.open(image_path).convert("L")  # 转换为灰度图像
    return np.array(img)

def compute_frequency_energy(image):
    """计算图像的频率能量分布"""
    # 应用快速傅里叶变换 (FFT)
    fft = np.fft.fft2(image)
    # 将零频率分量移到中心
    fft_shifted = np.fft.fftshift(fft)
    # 计算频率能量（幅值）
    magnitude_spectrum = np.abs(fft_shifted)
    return magnitude_spectrum

def normalize_frequency_coordinates(magnitude_spectrum):
    """归一化频率坐标到 [-0.5, 0.5] 范围"""
    h, w = magnitude_spectrum.shape
    y_freq = np.fft.fftshift(np.fft.fftfreq(h))  # 垂直方向频率坐标
    x_freq = np.fft.fftshift(np.fft.fftfreq(w))  # 水平方向频率坐标
    return y_freq, x_freq, magnitude_spectrum

from scipy.interpolate import RegularGridInterpolator

def interpolate_frequency_energy(y_freq, x_freq, magnitude_spectrum, target_grid_size=256):
    """将频域能量插值到目标网格上"""
    # 创建目标网格
    y_target = np.linspace(-0.5, 0.5, target_grid_size)
    x_target = np.linspace(-0.5, 0.5, target_grid_size)
    y_grid, x_grid = np.meshgrid(y_target, x_target, indexing="ij")
    
    # 创建插值器
    interpolator = RegularGridInterpolator((y_freq, x_freq), magnitude_spectrum, bounds_error=False, fill_value=0)
    
    # 插值到目标网格
    target_magnitude_spectrum = interpolator((y_grid, x_grid))
    
    return target_magnitude_spectrum

def compute_frequency_band_energy(magnitude_spectrum, num_bands=10):
    """计算多个频带的能量占比"""
    h, w = magnitude_spectrum.shape
    center_h, center_w = h // 2, w // 2
    
    # 初始化频带能量列表
    band_energies = []
    
    # 计算每个频带的能量
    for i in range(num_bands):
        # 计算当前频带的半径范围
        inner_radius = i * (min(h, w) // 2) // num_bands
        outer_radius = (i + 1) * (min(h, w) // 2) // num_bands
        
        # 创建一个掩码来提取当前频带的能量
        y, x = np.ogrid[-center_h:h - center_h, -center_w:w - center_w]
        mask = (x**2 + y**2 >= inner_radius**2) & (x**2 + y**2 < outer_radius**2)
        
        # 计算当前频带的能量
        band_energy = np.sum(magnitude_spectrum[mask])
        band_energies.append(band_energy)
    
    # 计算总能量
    total_energy = np.sum(band_energies)
    
    # 计算每个频带的能量占比
    band_ratios = [energy / total_energy for energy in band_energies]
    
    return band_ratios

def process_image(image_path, num_bands, target_grid_size):
    """处理单张图像并返回频带能量占比"""
    # 加载图像
    image = load_image(image_path)
    # 计算频率能量
    magnitude_spectrum = compute_frequency_energy(image)
    # 归一化频率坐标
    y_freq, x_freq, magnitude_spectrum = normalize_frequency_coordinates(magnitude_spectrum)
    # 插值频域能量到目标网格
    target_magnitude_spectrum = interpolate_frequency_energy(y_freq, x_freq, magnitude_spectrum, target_grid_size)
    # 计算频带能量占比
    band_ratios = compute_frequency_band_energy(target_magnitude_spectrum, num_bands)
    return band_ratios

def extract_frequency_features(dataset_dir, num_bands=20, target_grid_size=256):
    """提取数据集中所有图像的频带能量特征（多进程加速）"""
    dataset_dir = Path(dataset_dir)
    image_paths = list(dataset_dir.glob("**/*.jpg")) + list(dataset_dir.glob("**/*.png"))
    image_paths = image_paths[:5000]
    # 使用多进程加速
    num_processes = cpu_count()  # 获取 CPU 核心数
    with Pool(processes=num_processes) as pool:
        # 将任务分配到多个进程
        results = pool.starmap(
            process_image,
            [(image_path, num_bands, target_grid_size) for image_path in image_paths]
        )
    
    # 将结果转换为 NumPy 数组
    features = np.array(results)
    return features, image_paths

def cluster_images(features, n_clusters=5, use_pca=False):
    """对图像进行聚类（使用 MiniBatchKMeans 和降维）"""
    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 降维（可选）
    if use_pca:
        pca = PCA(n_components=10)  # 降到 10 维
        features_reduced = pca.fit_transform(features_scaled)
    else:
        features_reduced = features_scaled
    
    # 使用 MiniBatchKMeans 聚类
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000, random_state=42)
    labels = kmeans.fit_predict(features_reduced)
    
    # 可视化聚类结果（2D）
    if use_pca:
        pca_2d = PCA(n_components=2)
        features_2d = pca_2d.fit_transform(features_reduced)
    else:
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features_reduced)
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap="viridis", alpha=0.7)
    plt.colorbar(scatter, label="Cluster")
    plt.title("Clustering of Images Based on Frequency Band Energy")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()
    
    return labels

from multiprocessing import Pool

def copy_image(args):
    """复制单张图片到目标文件夹"""
    image_path, target_path = args
    shutil.copy(image_path, target_path)

def save_clustered_images_parallel(image_paths, labels, output_dir, num_processes=None):
    """
    使用多进程将图片按照聚类结果存储到对应的文件夹，并生成 CSV 文件。
    
    参数:
        image_paths (list): 图片路径列表。
        labels (list): 聚类标签列表。
        output_dir (str): 输出目录。
        num_processes (int): 进程数，默认为 CPU 核心数。
    """
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建聚类文件夹
    unique_labels = set(labels)
    for label in unique_labels:
        cluster_dir = output_dir / f"cluster_{label}"
        cluster_dir.mkdir(exist_ok=True)
    
    # 创建 CSV 文件
    csv_path = output_dir / "clustering_results.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Image Path", "Cluster Label"])  # 写入表头
        
        # 准备多进程任务
        tasks = []
        for image_path, label in zip(image_paths, labels):
            image_name = image_path.name
            target_path = output_dir / f"cluster_{label}" / image_name
            tasks.append((image_path, target_path))
            writer.writerow([str(target_path), label])
        
        # 使用多进程复制图片
        with Pool(processes=num_processes) as pool:
            pool.map(copy_image, tasks)
    
    print(f"聚类结果已保存到: {output_dir}")
    print(f"CSV 文件已生成: {csv_path}")


# 示例：提取特征并聚类
dataset_dir = "/nasdata2/private/zwlu/compress/naic2024/datasets/train"
num_bands = 50  # 使用 20 个频带
target_grid_size = 512  # 目标网格大小为 256x256

# 提取频带能量特征
features, image_paths = extract_frequency_features(dataset_dir, num_bands, target_grid_size)

# 对图像进行聚类
n_clusters = 10  # 聚类数量
labels = cluster_images(features, n_clusters, use_pca=True)

# 打印聚类结果
for i, (image_path, label) in enumerate(zip(image_paths, labels)):
    print(f"Image {i+1}: {image_path} -> Cluster {label}")
    
output_dir = "/nasdata2/private/zwlu/compress/naic2024/datasets/clustered_images"
save_clustered_images_parallel(image_paths, labels, output_dir, num_processes=cpu_count() - 1)
