#!/usr/bin/env python3
"""
绘制三个子图的综合对比图（使用 Matplotlib）
左：累积成功率 | 中：学习增益 | 右：稳定性损失
自动检索所有 JSON 文件，根据性能排序并分配颜色
"""

import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np

# 从命令行参数获取输入目录，默认为 outputs/cross
if len(sys.argv) > 1:
    cross_dir = Path(sys.argv[1])
else:
    cross_dir = Path("outputs/cross")

print(f"读取数据目录: {cross_dir}")

# 颜色方案：按性能从高到低分配（越好颜色越显眼）
COLOR_PALETTE = [
    "#FF0000",  # 红色 - 最好
    "#9B59B6",  # 紫色 - 第二
    "#E67E22",  # 橙色 - 第三
    "#2ECC71",  # 绿色 - 第四
    "#3498DB",  # 蓝色 - 第五
    "#F39C12",  # 黄色 - 第六
    "#1ABC9C",  # 青色 - 第七
    "#E74C3C",  # 深红色 - 第八
]


def extract_data(method_data):
    """从 JSON 数据中提取 cumulative_rates 和 total_counts
    优先从 overall 中读取，如果不存在则从顶层读取
    """
    # 优先从 overall 中读取
    if "overall" in method_data:
        overall = method_data.get("overall", {})
        cumulative_rates = overall.get("cumulative_rates", [])
        total_counts = overall.get("total_counts", [])
    else:
        # 从顶层读取
        cumulative_rates = method_data.get("cumulative_rates", [])
        total_counts = method_data.get("total_counts", [])

    return cumulative_rates, total_counts


# 读取所有 JSON 文件并计算最终成功率
data = {}
method_performance = {}
print(f"搜索 JSON 文件: {cross_dir}/*.json")
json_files = list(cross_dir.glob("*.json"))
print(f"找到 {len(json_files)} 个 JSON 文件")

for json_file in json_files:
    method_name = json_file.stem
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            method_data = json.load(f)
            data[method_name] = method_data

            # 获取最终成功率（用于排序）
            # 优先从 overall 中读取，如果不存在则从顶层读取
            if "overall" in method_data:
                overall = method_data.get("overall", {})
                final_rate = overall.get("final_success_rate", 0)
            else:
                final_rate = method_data.get("final_success_rate", 0)
            method_performance[method_name] = final_rate

            print(f"  ✓ 已加载 {method_name} (最终成功率: {final_rate:.4f})")
    except Exception as e:
        print(f"  ✗ 加载 {method_name} 失败: {e}")

# 按最终成功率从高到低排序
sorted_methods = sorted(method_performance.items(), key=lambda x: x[1], reverse=True)
print(f"\n性能排序（从高到低）:")
for rank, (method_name, final_rate) in enumerate(sorted_methods, 1):
    print(f"  {rank}. {method_name}: {final_rate:.4f} ({final_rate*100:.2f}%)")

# 创建方法配置（颜色根据性能分配）
methods_config = {}
for rank, (method_name, _) in enumerate(sorted_methods):
    color = COLOR_PALETTE[rank % len(COLOR_PALETTE)]
    # 使用文件名作为图例，首字母大写
    label = method_name.replace("-", " ").replace("_", " ").title()
    methods_config[method_name] = {"color": color, "label": label}

# 创建图形（1行3列）
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))

# 绘制顺序
plot_order = [method_name for method_name, _ in sorted_methods]

print(f"\n开始绘制三个子图")

# ============ 子图1: 累积成功率 ============
print("  绘制累积成功率...")
for method_name in plot_order:
    if method_name in data:
        method_data = data[method_name]
        cumulative_rates, total_counts = extract_data(method_data)

        if cumulative_rates and total_counts:
            config = methods_config[method_name]
            x = np.array(total_counts)
            y = np.array(cumulative_rates)
            ax1.plot(x, y, color=config["color"], label=config["label"], linewidth=1.5)

# 设置子图1
ax1.set_xlabel('Number of Samples', fontsize=12, fontweight='normal')
ax1.set_ylabel('Weighted Cumulative Success Rate', fontsize=12, fontweight='normal')
ax1.set_title('(a) Weighted Cumulative Success Rate', fontsize=14, fontweight='normal', pad=15)
ax1.grid(True, alpha=0.3, linestyle='-.', linewidth=0.8)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend(loc='right', fontsize=10, frameon=True, facecolor='white',
          edgecolor='gray', framealpha=0.9)
ax1.tick_params(labelsize=10)

# ============ 子图2: Learning Gain ============
print("  绘制 Learning Gain...")
for method_name in plot_order:
    if method_name in data:
        method_data = data[method_name]
        cumulative_rates, total_counts = extract_data(method_data)

        if cumulative_rates and total_counts:
            config = methods_config[method_name]

            # 计算 Learning Gain
            rates = np.array(cumulative_rates)
            learning_gain = []
            for t in range(len(rates)):
                future_max = np.max(rates[t:])
                lg_t = future_max - rates[t]
                learning_gain.append(lg_t)

            x = np.array(total_counts)
            y = np.array(learning_gain)
            ax2.plot(x, y, color=config["color"], label=config["label"], linewidth=1.5)

# 设置子图2
ax2.set_xlabel('Number of Samples', fontsize=12, fontweight='normal')
ax2.set_ylabel('Weighted Learning Gain', fontsize=12, fontweight='normal')
ax2.set_title('(b) Weighted Learning Gain', fontsize=14, fontweight='normal', pad=15)
ax2.set_ylim(bottom=0)
ax2.grid(True, alpha=0.3, linestyle='-.', linewidth=0.8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend(loc='right', fontsize=10, frameon=True, facecolor='white',
          edgecolor='gray', framealpha=0.9)
ax2.tick_params(labelsize=10)

# ============ 子图3: Stability Loss ============
print("  绘制 Stability Loss...")
for method_name in plot_order:
    if method_name in data:
        method_data = data[method_name]
        cumulative_rates, total_counts = extract_data(method_data)

        if cumulative_rates and total_counts:
            config = methods_config[method_name]

            # 计算 Stability Loss
            rates = np.array(cumulative_rates)
            stability_loss = []
            for t in range(len(rates)):
                historical_max = np.max(rates[:t+1])
                sl_t = historical_max - rates[t]
                stability_loss.append(sl_t)

            x = np.array(total_counts)
            y = np.array(stability_loss)
            ax3.plot(x, y, color=config["color"], label=config["label"], linewidth=1.5)

# 设置子图3
ax3.set_xlabel('Number of Samples', fontsize=12, fontweight='normal')
ax3.set_ylabel('Weighted Stability Loss', fontsize=12, fontweight='normal')
ax3.set_title('(c) Weighted Stability Loss', fontsize=14, fontweight='normal', pad=15)
ax3.set_ylim(bottom=0)
ax3.grid(True, alpha=0.3, linestyle='-.', linewidth=0.8)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.legend(loc='right', fontsize=10, frameon=True, facecolor='white',
          edgecolor='gray', framealpha=0.9)
ax3.tick_params(labelsize=10)

# 调整子图之间的间距
plt.tight_layout()

# 保存图片
output_path = cross_dir / "all_metrics.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n图片已保存到: {output_path}")

# 显示图表
plt.show()
