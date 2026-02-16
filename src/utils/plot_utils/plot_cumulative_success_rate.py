#!/usr/bin/env python3
"""
绘制 cross task 的累积成功率曲线图（使用 Matplotlib）
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
            # 兼容三种数据结构：
            # 1. 顶层直接包含数据
            # 2. 嵌套在 overall 键下
            # 3. mixed 任务：有 system_memory, personal_memory, 顶层 overall
            task_type = method_data.get("task_type", "")
            if task_type == "mixed":
                # 混合任务：使用顶层的 overall
                overall = method_data.get("overall", {})
                final_rate = overall.get("final_success_rate", 0)
            elif "overall" in method_data and "task_type" not in method_data:
                # 嵌套在 overall 键下（但不是 mixed 类型）
                overall = method_data.get("overall", {})
                final_rate = overall.get("final_success_rate", 0)
            else:
                # 顶层直接包含数据
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

# 创建图形（正方形）
fig, ax = plt.subplots(figsize=(7, 7))

# 绘制折线图（按性能从高到低的顺序）
plot_order = [method_name for method_name, _ in sorted_methods]

print(f"\n开始绘制")
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
            x = np.array(total_counts)
            y = np.array(cumulative_rates)

            # 绘制主线条
            ax.plot(x, y, color=config["color"], label=config["label"], linewidth=1.5)

            print(f"  ✓ {method_name} 绘制完成")

# 设置标签和标题
ax.set_xlabel('Number of Samples', fontsize=12, fontweight='normal')
ax.set_ylabel('Cumulative Success Rate', fontsize=12, fontweight='normal')
ax.set_title('Cumulative Success Rate (Cross Task)', fontsize=14, fontweight='normal', pad=15)

# 设置 y 轴范围
ax.set_ylim(0.2, 0.8)

# 添加点划线网格
ax.grid(True, alpha=0.3, linestyle='-.', linewidth=0.8)

# 去掉上边框和右边框（像参考图一样）
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 图例放在左下角
ax.legend(loc='right', fontsize=10, frameon=True, facecolor='white',
          edgecolor='gray', framealpha=0.9)

# 调整刻度字体大小
ax.tick_params(labelsize=10)

# 调整布局
plt.tight_layout()

# 保存图片
output_path = cross_dir / "cumulative_success_rate.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n图片已保存到: {output_path}")

# 显示图表
plt.show()
