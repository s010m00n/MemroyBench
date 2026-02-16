#!/usr/bin/env python3
"""
绘制每个方法的任务特定累积成功率曲线（使用 Matplotlib）
每个子图显示一个方法的 DB 和 OS 任务各自的累积成功率
- 当执行 DB 任务时，OS 的成功率保持不变
- 当执行 OS 任务时，DB 的成功率保持不变
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

# 颜色方案
DB_COLOR = "#E74C3C"  # 红色 - DB
OS_COLOR = "#3498DB"  # 蓝色 - OS


def extract_task_specific_rates(method_data):
    """从 JSON 数据中提取 DB 和 OS 任务各自的累积成功率"""
    # 获取 detailed_results
    detailed_results = method_data.get("detailed_results", [])

    if not detailed_results:
        return None, None, None, None

    # 初始化
    db_success = 0
    db_total = 0
    os_success = 0
    os_total = 0

    db_rates = []
    os_rates = []
    db_x = []  # DB 的 x 坐标（总样本数）
    os_x = []  # OS 的 x 坐标（总样本数）

    for i, item in enumerate(detailed_results, 1):
        task = item.get("task", "")
        success = item.get("success", False)

        # 判断任务类型
        is_db = "db" in task.lower()

        if is_db:
            # DB 任务
            db_total += 1
            if success:
                db_success += 1
            db_rate = db_success / db_total if db_total > 0 else 0
            db_rates.append(db_rate)
            db_x.append(i)
            # OS 保持不变（添加重复值以形成阶梯）
            if os_rates:
                os_rates.append(os_rates[-1])
                os_x.append(i)
        else:
            # OS 任务
            os_total += 1
            if success:
                os_success += 1
            os_rate = os_success / os_total if os_total > 0 else 0
            os_rates.append(os_rate)
            os_x.append(i)
            # DB 保持不变（添加重复值以形成阶梯）
            if db_rates:
                db_rates.append(db_rates[-1])
                db_x.append(i)

    return db_x, db_rates, os_x, os_rates


# 读取所有 JSON 文件
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
            task_type = method_data.get("task_type", "")
            if task_type == "mixed":
                overall = method_data.get("overall", {})
                final_rate = overall.get("final_success_rate", 0)
            elif "overall" in method_data and "task_type" not in method_data:
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

# 只取前4个方法绘制
top_methods = [method_name for method_name, _ in sorted_methods[:4]]

# 创建图形（2行2列）
fig, axes = plt.subplots(2, 2, figsize=(14, 14))
axes = axes.flatten()

print(f"\n开始绘制任务特定曲线")

for idx, method_name in enumerate(top_methods):
    if method_name in data:
        method_data = data[method_name]

        # 提取任务特定的成功率
        db_x, db_rates, os_x, os_rates = extract_task_specific_rates(method_data)

        if db_x is None or os_x is None:
            print(f"  ✗ {method_name} 没有 detailed_results 数据")
            continue

        ax = axes[idx]

        # 绘制 DB 曲线
        if db_x and db_rates:
            ax.plot(db_x, db_rates, color=DB_COLOR, label='DB Task', linewidth=2, linestyle='-')

        # 绘制 OS 曲线
        if os_x and os_rates:
            ax.plot(os_x, os_rates, color=OS_COLOR, label='OS Task', linewidth=2, linestyle='-')

        # 设置子图标题和标签
        label = method_name.replace("-", " ").replace("_", " ").title()
        ax.set_title(f'{label}', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Number of Samples', fontsize=12, fontweight='normal')
        ax.set_ylabel('Cumulative Success Rate', fontsize=12, fontweight='normal')

        # 设置 y 轴范围
        ax.set_ylim(0, 1.0)

        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='-.', linewidth=0.8)

        # 去掉上边框和右边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 添加图例
        ax.legend(loc='lower right', fontsize=11, frameon=True, facecolor='white',
                 edgecolor='gray', framealpha=0.9)

        # 调整刻度字体大小
        ax.tick_params(labelsize=10)

        print(f"  ✓ {method_name} 绘制完成")

# 隐藏多余的子图（如果方法少于4个）
for idx in range(len(top_methods), 4):
    axes[idx].axis('off')

# 调整子图之间的间距
plt.tight_layout()

# 保存图片
output_path = cross_dir / "task_specific_curves.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n图片已保存到: {output_path}")

# 显示图表
plt.show()
