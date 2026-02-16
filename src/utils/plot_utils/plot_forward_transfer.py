#!/usr/bin/env python3
"""
绘制同环境前向迁移（Within-Environment Forward Transfer）曲线
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    import statistics


def load_sample_result(file_path: Path) -> Dict:
    """读取单个样本的结果文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    except Exception as e:
        print(f"Warning: Failed to load {file_path}: {e}")
        return None


def get_performance(result: Dict, metric_type: str = "llm_score") -> float:
    """从结果中提取指定类型的指标"""
    if not result:
        return None

    try:
        # 可能在 result.result.metrics 或 result.metrics 中
        if "result" in result and "result" in result["result"]:
            metrics = result["result"]["result"].get("metrics", {})
        elif "result" in result:
            metrics = result["result"].get("metrics", {})
        else:
            metrics = result.get("metrics", {})

        return metrics.get(metric_type, 0.0)
    except Exception as e:
        print(f"Warning: Failed to extract metric: {e}")
        return None


def calculate_forward_transfer(transfer_dir: Path, metric_type: str = "llm_score") -> Tuple[List[float], Dict]:
    """
    计算同环境前向迁移指标

    Args:
        transfer_dir: transfer目录路径 (e.g., outputs/transfer/in-environment/mem0/seed66-1)
        metric_type: 指标类型 (llm_score, f1_score, bleu_score)

    Returns:
        (ftg_list, stats): FTG列表和统计信息
    """
    transfer_dir = Path(transfer_dir)
    train_dir = transfer_dir / "transfer_train"
    forward_test_dir = transfer_dir / "forward_transfer_test"

    # 1. 读取所有训练样本的立即测试性能 P_immediate
    print(f"\n{'='*60}")
    print(f"处理目录: {transfer_dir}")
    print(f"\n读取训练样本的立即测试性能...")

    immediate_performance = {}  # {(task, index): performance}

    for task_dir in train_dir.iterdir():
        if task_dir.is_dir():
            task_name = task_dir.name
            for sample_file in task_dir.glob("*.json"):
                index = int(sample_file.stem)
                result = load_sample_result(sample_file)
                if result:
                    perf = get_performance(result, metric_type)
                    if perf is not None:
                        immediate_performance[(task_name, index)] = perf

    print(f"共读取 {len(immediate_performance)} 个训练样本的立即测试性能")

    # 2. 读取前向迁移测试结果并计算FTG
    print(f"\n读取前向迁移测试结果...")

    ftg_list = []
    forward_data = []  # 保存详细数据用于分析

    for task_dir in forward_test_dir.iterdir():
        if task_dir.is_dir():
            task_name = task_dir.name

            for sample_file in task_dir.glob("*.json"):
                # 文件名格式: trainX_testY.json
                filename = sample_file.stem
                if not filename.startswith("train"):
                    continue

                parts = filename.split("_")
                train_idx = int(parts[0].replace("train", ""))
                test_idx = int(parts[1].replace("test", ""))

                # 读取前向测试结果
                result = load_sample_result(sample_file)
                if not result:
                    continue

                p_forward = get_performance(result, metric_type)
                if p_forward is None:
                    continue

                # 获取训练样本的立即测试性能
                p_immediate_train = immediate_performance.get((task_name, train_idx))
                if p_immediate_train is None or p_immediate_train == 0:
                    continue

                # 获取测试样本的立即测试性能（作为baseline）
                p_immediate_test = immediate_performance.get((task_name, test_idx))
                if p_immediate_test is None:
                    continue

                # 计算FTG
                # FTG = (P_forward - P_immediate_test) / P_immediate_train * 100%
                ftg = (p_forward - p_immediate_test) / p_immediate_train * 100.0
                ftg_list.append(ftg)

                forward_data.append({
                    "task": task_name,
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                    "p_immediate_train": p_immediate_train,
                    "p_forward": p_forward,
                    "p_immediate_test": p_immediate_test,
                    "ftg": ftg
                })

    # 3. 计算统计信息
    if ftg_list:
        if MATPLOTLIB_AVAILABLE:
            avg_ftg = np.mean(ftg_list)
            median_ftg = np.median(ftg_list)
            std_ftg = np.std(ftg_list)
        else:
            avg_ftg = statistics.mean(ftg_list)
            median_ftg = statistics.median(ftg_list)
            std_ftg = statistics.stdev(ftg_list) if len(ftg_list) > 1 else 0.0

        print(f"\n{'='*60}")
        print(f"前向迁移统计:")
        print(f"  样本数: {len(ftg_list)}")
        print(f"  平均FTG: {avg_ftg:.2f}%")
        print(f"  中位数FTG: {median_ftg:.2f}%")
        print(f"  标准差: {std_ftg:.2f}%")
        print(f"  最小值: {min(ftg_list):.2f}%")
        print(f"  最大值: {max(ftg_list):.2f}%")

        stats = {
            "metric_type": metric_type,
            "num_samples": len(ftg_list),
            "avg_ftg": avg_ftg,
            "median_ftg": median_ftg,
            "std_ftg": std_ftg,
            "min_ftg": min(ftg_list),
            "max_ftg": max(ftg_list),
            "forward_data": forward_data
        }
    else:
        print("\nWarning: No valid forward transfer data found!")
        stats = {
            "metric_type": metric_type,
            "num_samples": 0
        }

    return ftg_list, stats


def plot_forward_transfer_comparison(results: Dict[str, Tuple[List[float], Dict]],
                                     output_path: Path = None,
                                     metric_type: str = "llm_score"):
    """
    绘制多个方法的前向迁移折线图

    Args:
        results: {method_name: (ftg_list, stats)}
        output_path: 输出图片路径
        metric_type: 指标类型
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping plot generation")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # 颜色映射
    colors = {
        "mem0": "#e74c3c",
        "streamICL": "#3498db",
        "awmPro": "#2ecc71"
    }
    markers = {
        "mem0": "o",
        "streamICL": "s",
        "awmPro": "^"
    }

    # 绘制FTG折线图（只显示移动平均）
    for method_name, (ftg_list, stats) in results.items():
        if ftg_list and "forward_data" in stats:
            forward_data = stats["forward_data"]
            # 按训练样本索引排序
            sorted_data = sorted(forward_data, key=lambda x: x["train_idx"])

            train_indices = [d["train_idx"] for d in sorted_data]
            ftg_values = [d["ftg"] for d in sorted_data]

            color = colors.get(method_name, None)
            marker = markers.get(method_name, "o")

            # 移动平均线
            window_size = 10
            if len(ftg_values) >= window_size:
                moving_avg = np.convolve(ftg_values, np.ones(window_size)/window_size, mode='valid')
                x_vals = train_indices[window_size-1:]
            else:
                moving_avg = ftg_values
                x_vals = train_indices

            ax.plot(x_vals, moving_avg,
                   label=f"{method_name} (avg: {stats['avg_ftg']:.1f}%)",
                   color=color, marker=marker,
                   linewidth=2.5, markersize=6, alpha=0.9,
                   markevery=max(1, len(x_vals)//15))

    ax.set_xlabel('Training Sample Index', fontsize=13)
    ax.set_ylabel('Forward Transfer Gain (%)', fontsize=13)
    ax.set_title(f'Within-Environment Forward Transfer ({metric_type})',
                fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n图表已保存到: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="计算和绘制同环境前向迁移指标")
    parser.add_argument("base_dir", type=str, nargs="?", default=None,
                       help="基础目录路径 (如 outputs/transfer/in-environment)，会自动扫描所有子方法")
    parser.add_argument("--methods", nargs="*", type=str, default=None,
                       help="指定方法名列表 (可选，不指定则扫描所有)")
    parser.add_argument("--metric", type=str, default="llm_score",
                       choices=["llm_score", "f1_score", "bleu_score"],
                       help="指标类型 (默认: llm_score)")
    parser.add_argument("--output", type=str, default=None,
                       help="输出图片路径 (可选，默认保存到基础目录)")
    parser.add_argument("--save-stats", action="store_true",
                       help="保存统计信息到JSON文件")

    args = parser.parse_args()

    # 如果未指定base_dir，使用默认路径
    if args.base_dir is None:
        args.base_dir = "outputs/transfer/in-environment"

    base_path = Path(args.base_dir)

    # 自动扫描所有方法目录
    if args.methods:
        # 使用指定的方法列表
        transfer_dirs = []
        for method in args.methods:
            method_dir = base_path / method
            # 查找seed目录
            seed_dirs = list(method_dir.glob("seed*"))
            if seed_dirs:
                transfer_dirs.append(seed_dirs[0])  # 使用第一个seed目录
            else:
                print(f"Warning: No seed directory found for method {method}")
    else:
        # 扫描所有方法
        transfer_dirs = []
        for method_dir in base_path.iterdir():
            if method_dir.is_dir():
                # 查找seed目录
                seed_dirs = list(method_dir.glob("seed*"))
                if seed_dirs:
                    transfer_dirs.append(seed_dirs[0])

    if not transfer_dirs:
        print(f"Error: No valid transfer directories found in {base_path}")
        return

    print(f"\n找到 {len(transfer_dirs)} 个方法目录:")
    for d in transfer_dirs:
        print(f"  - {d.parent.name}/{d.name}")

    # 处理每个目录
    results = {}
    for transfer_path in transfer_dirs:
        method_name = transfer_path.parent.name  # 从路径提取方法名

        ftg_list, stats = calculate_forward_transfer(transfer_path, args.metric)
        results[method_name] = (ftg_list, stats)

        # 保存统计信息
        if args.save_stats:
            stats_file = transfer_path / f"forward_transfer_stats_{args.metric}.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                # 移除forward_data以减少文件大小
                save_stats = {k: v for k, v in stats.items() if k != "forward_data"}
                json.dump(save_stats, f, indent=2, ensure_ascii=False)
            print(f"统计信息已保存到: {stats_file}")

    # 绘制对比图
    if len(results) > 0:
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = base_path / f"forward_transfer_comparison_{args.metric}.png"

        plot_forward_transfer_comparison(results, output_path, args.metric)


if __name__ == "__main__":
    main()
