import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from scipy.interpolate import griddata
from multiprocessing import Pool, cpu_count

import tqdm

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

def analyze_dataset_frequency_bands(dataset_dir, num_bands=10, target_grid_size=256):
    """分析整个数据集的频带能量分布（多进程加速）"""
    dataset_dir = Path(dataset_dir)
    image_paths = list(dataset_dir.glob("**/*.jpg")) + list(dataset_dir.glob("**/*.png"))
    print(len(image_paths))
    random.shuffle(image_paths)
    # 初始化频带能量累加器
    total_band_ratios = np.zeros(num_bands)
    
    # 使用多进程加速
    num_processes = cpu_count()  # 获取 CPU 核心数
    with Pool(processes=num_processes) as pool:
        # 将任务分配到多个进程
        results = pool.starmap(
            process_image,
            [(image_path, num_bands, target_grid_size) for image_path in image_paths[:100]]
        )
    
    # 合并结果
    for band_ratios in tqdm.tqdm(results):
        total_band_ratios += np.array(band_ratios)
    
    # 计算平均频带能量占比
    avg_band_ratios = total_band_ratios / len(image_paths)
    
    # 绘制频带能量分布图
    plot_frequency_band_distribution(avg_band_ratios)

def plot_frequency_band_distribution(band_ratios):
    """绘制频带能量分布图"""
    num_bands = len(band_ratios)
    bands = np.arange(num_bands)
    
    plt.figure(figsize=(10, 5))
    plt.bar(bands, band_ratios, color='blue', alpha=0.7)
    plt.xlabel("Frequency Band")
    plt.ylabel("Energy Ratio")
    plt.title("Frequency Band Energy Distribution")
    plt.xticks(bands, [f"Band {i+1}" for i in bands])
    plt.grid(True)
    plt.show()

# if __name__ == "__main__":
# 示例：分析整个数据集的频带能量分布
dataset_dir = "/nasdata2/private/zwlu/compress/naic2024/datasets/train"
analyze_dataset_frequency_bands(dataset_dir, num_bands=20, target_grid_size=256)  # 使用 20 个频带，目标网格大小为 256x256
