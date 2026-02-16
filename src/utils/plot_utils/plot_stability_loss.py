#!/usr/bin/env python3
"""
绘制 Stability Loss 曲线图（使用 Matplotlib）
Stability Loss: SL_t = P_max^t - P_t (历史最高点 - 当前值)
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

# 读取所有 JSON 文件并计算平均 Stability Loss
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

            # 计算 Stability Loss 用于排序（平均值，越小越好）
            # 兼容三种数据结构
            task_type = method_data.get("task_type", "")
            if task_type == "mixed":
                # 混合任务：使用顶层的 overall
                overall = method_data.get("overall", {})
                cumulative_rates = overall.get("cumulative_rates", [])
            elif "overall" in method_data and "task_type" not in method_data:
                # 嵌套在 overall 键下（但不是 mixed 类型）
                overall = method_data.get("overall", {})
                cumulative_rates = overall.get("cumulative_rates", [])
            else:
                # 顶层直接包含数据
                cumulative_rates = method_data.get("cumulative_rates", [])

            if cumulative_rates:
                rates = np.array(cumulative_rates)
                stability_loss_values = []
                for t in range(len(rates)):
                    historical_max = np.max(rates[:t+1])
                    sl_t = historical_max - rates[t]
                    stability_loss_values.append(sl_t)
                avg_sl = np.mean(stability_loss_values)
                method_performance[method_name] = avg_sl
                print(f"  ✓ 已加载 {method_name} (平均 SL: {avg_sl:.4f})")
            else:
                method_performance[method_name] = float('inf')
                print(f"  ✓ 已加载 {method_name} (无数据)")
    except Exception as e:
        print(f"  ✗ 加载 {method_name} 失败: {e}")

# 按平均 Stability Loss 从低到高排序（越小越好，说明越稳定）
sorted_methods = sorted(method_performance.items(), key=lambda x: x[1])
print(f"\n性能排序（SL 从低到高，越低越好）:")
for rank, (method_name, avg_sl) in enumerate(sorted_methods, 1):
    if avg_sl == float('inf'):
        print(f"  {rank}. {method_name}: 无数据")
    else:
        print(f"  {rank}. {method_name}: {avg_sl:.4f}")

# 创建方法配置（颜色根据性能分配）
methods_config = {}
for rank, (method_name, _) in enumerate(sorted_methods):
    color = COLOR_PALETTE[rank % len(COLOR_PALETTE)]
    # 使用文件名作为图例，首字母大写
    label = method_name.replace("-", " ").replace("_", " ").title()
    methods_config[method_name] = {"color": color, "label": label}

# 创建图形（正方形）
fig, ax = plt.subplots(figsize=(7, 7))

# 绘制折线图（按性能从高到低的顺序）
plot_order = [method_name for method_name, _ in sorted_methods]

print(f"\n开始绘制 Stability Loss")
for method_name in plot_order:
    if method_name in data:
        method_data = data[method_name]

        # 兼容三种数据结构
        task_type = method_data.get("task_type", "")
        if task_type == "mixed":
            # 混合任务：使用顶层的 overall
            overall = method_data.get("overall", {})
            cumulative_rates = overall.get("cumulative_rates", [])
            total_counts = overall.get("total_counts", [])
        elif "overall" in method_data and "task_type" not in method_data:
            # 嵌套在 overall 键下（但不是 mixed 类型）
            overall = method_data.get("overall", {})
            cumulative_rates = overall.get("cumulative_rates", [])
            total_counts = overall.get("total_counts", [])
        else:
            # 顶层直接包含数据
            cumulative_rates = method_data.get("cumulative_rates", [])
            total_counts = method_data.get("total_counts", [])

        if cumulative_rates and total_counts:
            config = methods_config[method_name]

            # 计算 Stability Loss: SL_t = P_max^t - P_t
            rates = np.array(cumulative_rates)
            stability_loss = []

            for t in range(len(rates)):
                # 历史最高点：从开始到当前时刻 t 的最大值
                historical_max = np.max(rates[:t+1])
                sl_t = historical_max - rates[t]
                stability_loss.append(sl_t)

            x = np.array(total_counts)
            y = np.array(stability_loss)

            # 绘制主线条
            ax.plot(x, y, color=config["color"], label=config["label"], linewidth=1.5)

            print(f"  ✓ {method_name} 绘制完成")

# 设置标签和标题
ax.set_xlabel('Number of Samples', fontsize=12, fontweight='normal')
ax.set_ylabel('Stability Loss', fontsize=12, fontweight='normal')
ax.set_title('Stability Loss (Cross Task)', fontsize=14, fontweight='normal', pad=15)

# 设置 y 轴范围（从 0 开始）
ax.set_ylim(bottom=0)

# 添加点划线网格
ax.grid(True, alpha=0.3, linestyle='-.', linewidth=0.8)

# 去掉上边框和右边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 图例放在右上角
ax.legend(loc='upper right', fontsize=10, frameon=True, facecolor='white',
          edgecolor='gray', framealpha=0.9)

# 调整刻度字体大小
ax.tick_params(labelsize=10)

# 调整布局
plt.tight_layout()

# 保存图片
output_path = cross_dir / "stability_loss.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n图片已保存到: {output_path}")

# 显示图表
plt.show()
