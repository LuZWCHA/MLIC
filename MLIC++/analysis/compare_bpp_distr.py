import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取两个 CSV 文件
file1 = "/nasdata2/private/zwlu/compress/naic2024/MLIC/MLIC++/playground/experiments/mlicpp_small_dec_mse_q3_finetune_ddp_trainer_exp/statistics_epoch0.csv"  # 替换为第一个文件路径
file2 = "/nasdata2/private/zwlu/compress/naic2024/MLIC/MLIC++/playground/experiments/mlicpp_small_dec_mse_q3_finetune_ddp_trainer_exp_testset/statistics_epoch0.csv"  # 替换为第二个文件路径

data1 = pd.read_csv(file1)
data2 = pd.read_csv(file2)

# 提取 bpp 列
bpp_data1 = data1["bpp"]
bpp_data2 = data2["bpp"]

# 计算 bins 的范围
bins = np.linspace(
    min(bpp_data1.min(), bpp_data2.min()),  # 取两个数据集的最小值
    max(bpp_data1.max(), bpp_data2.max()),  # 取两个数据集的最大值
    num=11  # 10 bins 需要 11 个边界
)

# 计算直方图
counts1, bin_edges = np.histogram(bpp_data1, bins=bins)
counts2, _ = np.histogram(bpp_data2, bins=bins)

# 归一化直方图
counts1 = counts1 / counts1.sum()  # 归一化到 0-1
counts2 = counts2 / counts2.sum()  # 归一化到 0-1

# 计算每个 bin 的中间值
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# 设置柱子宽度和间距
bar_width = (bin_edges[1] - bin_edges[0]) * 0.4  # 柱子宽度为 bin 宽度的 40%
offset = bar_width / 2  # 偏移量，使两个柱子并排显示

# 绘制柱状图
plt.figure(figsize=(10, 6))
plt.bar(bin_centers - offset, counts1, width=bar_width, edgecolor="black", alpha=0.7, label="train")
plt.bar(bin_centers + offset, counts2, width=bar_width, edgecolor="black", alpha=0.7, label="val")

# 设置标题和标签
plt.title("Normalized BPP Distribution Comparison", fontsize=16)
plt.xlabel("BPP (Bits Per Pixel)", fontsize=14)
plt.ylabel("Normalized Frequency", fontsize=14)
plt.legend(fontsize=12)  # 显示图例
plt.grid(axis="y", linestyle="--", alpha=0.7)

# 设置 x 轴刻度为 bin 的中间值
plt.xticks(bin_centers, [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)], rotation=45, fontsize=10)

# 显示图形
plt.tight_layout()
plt.show()