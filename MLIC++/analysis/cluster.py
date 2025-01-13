import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool, cpu_count

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

def interpolate_frequency_energy(y_freq, x_freq, magnitude_spectrum, target_grid_size=256):
    """将频域能量插值到目标网格上"""
    # 创建目标网格
    y_target = np.linspace(-0.5, 0.5, target_grid_size)
    x_target = np.linspace(-0.5, 0.5, target_grid_size)
    y_grid, x_grid = np.meshgrid(y_target, x_target, indexing="ij")
    
    # 创建原始网格
    y_original, x_original = np.meshgrid(y_freq, x_freq, indexing="ij")
    
    # 将原始频域能量插值到目标网格
    points = np.column_stack((y_original.ravel(), x_original.ravel()))
    values = magnitude_spectrum.ravel()
    target_magnitude_spectrum = griddata(points, values, (y_grid, x_grid), method="cubic", fill_value=0)
    
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

def extract_frequency_features(dataset_dir, num_bands=10, target_grid_size=256):
    """提取数据集中所有图像的频带能量特征（多进程加速）"""
    dataset_dir = Path(dataset_dir)
    image_paths = list(dataset_dir.glob("**/*.jpg")) + list(dataset_dir.glob("**/*.png"))
    
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

def cluster_images(features, n_clusters=5, use_pca=True):
    """对图像进行聚类（使用 MiniBatchKMeans 和降维）"""
    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 降维（可选）
    if use_pca:
        pca = PCA(n_components=50)  # 降到 50 维
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

# 示例：提取特征并聚类
dataset_dir = "path/to/your/dataset"
num_bands = 20  # 使用 20 个频带
target_grid_size = 256  # 目标网格大小为 256x256

# 提取频带能量特征
features, image_paths = extract_frequency_features(dataset_dir, num_bands, target_grid_size)

# 对图像进行聚类
n_clusters = 5  # 聚类数量
labels = cluster_images(features, n_clusters, use_pca=True)

# 打印聚类结果
for i, (image_path, label) in enumerate(zip(image_paths, labels)):
    print(f"Image {i+1}: {image_path} -> Cluster {label}")
