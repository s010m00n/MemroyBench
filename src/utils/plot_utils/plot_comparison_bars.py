#!/usr/bin/env python3
"""
绘制 streamICL 方法相对于 zero-shot 的性能提升柱状图
每个任务一组，显示 front 和 tail 两种方法相对于 zero-shot 的 SR 提升
"""

import matplotlib.pyplot as plt
import numpy as np

# 原始数据（从表格中提取）
data = {
    "zero_shot": {
        "dbbench": {"SR": 49.1, "AS": 2.78},
        "os": {"SR": 38.67, "AS": 2.55},
        "alfworld": {"SR": 17.4, "AS": 18.45}
    },
    "streamICL_front": {
        "dbbench": {"SR": 58.7, "AS": 2.65},
        "os": {"SR": 24.3, "AS": 4.36},
        "alfworld": {"SR": 32.1, "AS": 16.43}
    },
    "streamICL_tail": {
        "dbbench": {"SR": 56.7, "AS": 2.64},
        "os": {"SR": 22.2, "AS": 3.72},
        "alfworld": {"SR": 14.7, "AS": 18.06}
    }
}

# 任务列表
tasks = ["dbbench", "os", "alfworld"]
task_labels = ["DBBench", "OS", "ALFWorld"]

# 计算相对于 zero_shot 的提升
improvements_front = []
improvements_tail = []

for task in tasks:
    baseline = data["zero_shot"][task]["SR"]
    front_sr = data["streamICL_front"][task]["SR"]
    tail_sr = data["streamICL_tail"][task]["SR"]

    improvements_front.append(front_sr - baseline)
    improvements_tail.append(tail_sr - baseline)

    print(f"{task}:")
    print(f"  StreamICL (front): {front_sr:.2f}% vs {baseline:.2f}% = {front_sr - baseline:+.2f}%")
    print(f"  StreamICL (tail):  {tail_sr:.2f}% vs {baseline:.2f}% = {tail_sr - baseline:+.2f}%")

# 创建柱状图
fig, ax = plt.subplots(figsize=(10, 6))

# 设置柱子的位置
x = np.arange(len(tasks))
width = 0.35

# 绘制柱状图
bars1 = ax.bar(x - width/2, improvements_front, width,
               label='StreamICL (In-between)', color='#3498DB', edgecolor='black', linewidth=0.8)
bars2 = ax.bar(x + width/2, improvements_tail, width,
               label='StreamICL (Last)', color='#E67E22', edgecolor='black', linewidth=0.8)

# 在柱子上添加数值标签
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        # 根据正负值调整标签位置
        va = 'bottom' if height >= 0 else 'top'
        y_offset = 0.5 if height >= 0 else -0.5

        ax.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                f'{height:+.1f}',
                ha='center', va=va, fontsize=10, fontweight='bold')

add_value_labels(bars1)
add_value_labels(bars2)

# 设置标签和标题
ax.set_xlabel('Task', fontsize=12, fontweight='normal')
ax.set_ylabel('SR Improvement over Zero-Shot (%)', fontsize=12, fontweight='normal')
ax.set_title('StreamICL Performance Improvement over Zero-Shot', fontsize=14, fontweight='bold', pad=15)

# 设置 x 轴刻度
ax.set_xticks(x)
ax.set_xticklabels(task_labels, fontsize=11)

# 添加水平零线
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

# 添加网格
ax.grid(True, axis='y', alpha=0.3, linestyle='-.', linewidth=0.8)

# 去掉上边框和右边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 图例
ax.legend(loc='lower right', fontsize=11, frameon=True, facecolor='white',
          edgecolor='gray', framealpha=0.9)

# 调整刻度字体大小
ax.tick_params(labelsize=10)

# 调整布局
plt.tight_layout()

# 保存图片
output_path = "comparison_bars.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n图片已保存到: {output_path}")

# 显示图表
plt.show()
