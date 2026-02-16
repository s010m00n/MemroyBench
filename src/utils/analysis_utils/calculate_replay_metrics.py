#!/usr/bin/env python3
"""
计算 Replay 模式下的两个核心指标:
1. Average Success Rate (平均成功率): 所有 replay 阶段测试的平均性能
2. Forgetting Gain (遗忘度): 量化从学习完成到后续 replay 的性能下降

Usage:
    python calculate_replay_metrics.py <replay_dir>

Example:
    python calculate_replay_metrics.py outputs/replay/mem0/seed66-1
"""

import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple
from collections import defaultdict


def load_sample_result(file_path: Path) -> Dict:
    """加载单个样本的结果文件"""
    if not file_path.exists():
        return None

    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_performance(result: Dict, metric_type: str = "llm_score") -> float:
    """
    从结果中提取性能指标

    Args:
        result: 样本结果字典
        metric_type: 指标类型，可选 "llm_score", "f1_score", "bleu_score"

    Returns:
        性能分数 (0-1)
    """
    if result is None:
        return 0.0

    metrics = result.get("result", {}).get("metrics", {})

    # 尝试从 history 中获取 reward (对于 train immediate test)
    if "reward" in result.get("history", [{}])[-1]:
        return result["history"][-1]["reward"]

    # 从 metrics 中获取指标
    if metric_type == "llm_score":
        return metrics.get("llm_score", 0.0)
    elif metric_type == "f1_score":
        return metrics.get("f1_score", 0.0)
    elif metric_type == "bleu_score":
        return metrics.get("bleu_score", 0.0)
    else:
        # 默认使用三个指标的平均值
        f1 = metrics.get("f1_score", 0.0)
        bleu = metrics.get("bleu_score", 0.0)
        llm = metrics.get("llm_score", 0.0)
        return (f1 + bleu + llm) / 3.0


def calculate_replay_metrics(
    replay_dir: Path,
    metric_type: str = "llm_score"
) -> Dict:
    """
    计算 Replay 模式的两个核心指标

    Args:
        replay_dir: replay 实验目录 (如 outputs/replay/mem0/seed66-1)
        metric_type: 性能指标类型

    Returns:
        包含 Average Success Rate 和 Forgetting Gain 的字典
    """
    print(f"读取 Replay 数据目录: {replay_dir}")

    # 1. 找出所有 replay 阶段
    replay_stages = sorted([d for d in replay_dir.iterdir() if d.is_dir() and d.name.startswith("replay")])
    K = len(replay_stages)
    print(f"找到 {K} 个 replay 阶段: {[s.name for s in replay_stages]}")

    # 2. 数据结构存储
    # immediate_performance[stage_idx][(task, index)] = performance
    immediate_performance = {}
    # replay_performance[stage_idx][(task, index)] = performance
    replay_performance = defaultdict(dict)
    # stage_samples[stage_idx] = set of (task, index) learned in this stage
    stage_samples = defaultdict(set)

    # 3. 先读取主目录下 test/ 中的所有立即测试结果
    immediate_test_dir = replay_dir / "test"
    print(f"\n读取立即测试结果 (主目录 test/):")
    if immediate_test_dir.exists():
        for task_dir in immediate_test_dir.iterdir():
            if task_dir.is_dir():
                task_name = task_dir.name
                for sample_file in task_dir.glob("*.json"):
                    index = int(sample_file.stem)
                    result = load_sample_result(sample_file)
                    if result and result.get("split") == "immediate_test":
                        perf = get_performance(result, metric_type)
                        sample_key = (task_name, index)
                        # 将立即测试结果存储（暂不知道是在哪个replay阶段学习的）
                        if "immediate_all" not in immediate_performance:
                            immediate_performance["immediate_all"] = {}
                        immediate_performance["immediate_all"][sample_key] = perf
                        print(f"  {task_name}/{index}: immediate={perf:.4f}")

    print(f"\n总共 {len(immediate_performance.get('immediate_all', {}))} 个样本的立即测试结果")

    # 4. 从主目录的execution_order确定每个样本是在哪个replay阶段学习的
    main_exec_order = replay_dir / "execution_order.json"
    sample_to_replay_stage = {}  # (task, index) -> replay_stage_idx

    if main_exec_order.exists():
        with open(main_exec_order, 'r', encoding='utf-8') as f:
            exec_data = json.load(f)
            # 假设每个replay阶段学习m个样本，通过execution_order推断
            # 先读取replay1/test/execution_order看有多少个样本
            replay1_test_exec = replay_stages[0] / "test" / "execution_order.json"
            if replay1_test_exec.exists():
                with open(replay1_test_exec, 'r', encoding='utf-8') as rf:
                    first_replay_samples = json.load(rf)
                    samples_per_stage = len(first_replay_samples)
                    print(f"\n每个replay阶段约学习 {samples_per_stage} 个样本")

                    # 根据execution_order分配样本到各阶段
                    for i, item in enumerate(exec_data):
                        if item.get("split") == "train":
                            stage_idx = (i // samples_per_stage) + 1
                            sample_key = (item["task"], item["index"])
                            sample_to_replay_stage[sample_key] = stage_idx
                            stage_samples[stage_idx].add(sample_key)

    print(f"\n成功映射 {len(sample_to_replay_stage)} 个样本到各replay阶段")

    # 5. 读取每个 replay 阶段的重放测试结果
    for stage_idx, stage_dir in enumerate(replay_stages, start=1):
        print(f"\n处理阶段 {stage_idx}: {stage_dir.name}")

        # 5.1 读取 test 中的 replay 测试结果
        test_dir = stage_dir / "test"
        if test_dir.exists():
            for task_dir in test_dir.iterdir():
                if task_dir.is_dir():
                    task_name = task_dir.name
                    for sample_file in task_dir.glob("*.json"):
                        index = int(sample_file.stem)
                        result = load_sample_result(sample_file)
                        if result and result.get("split") == "test":
                            perf = get_performance(result, metric_type)
                            sample_key = (task_name, index)
                            replay_performance[stage_idx][sample_key] = perf
                            print(f"  Replay测试 - {task_name}/{index}: replay={perf:.4f}")

    # 6. 计算 Average Success Rate
    all_replay_scores = []
    for stage_idx in replay_performance:
        all_replay_scores.extend(replay_performance[stage_idx].values())

    avg_success_rate = sum(all_replay_scores) / len(all_replay_scores) if all_replay_scores else 0.0

    print(f"\n{'='*60}")
    print(f"Average Success Rate: {avg_success_rate:.4f} ({avg_success_rate*100:.2f}%)")
    print(f"  (基于 {len(all_replay_scores)} 次 replay 测试)")

    # 7. 计算 Forgetting Gain
    # FG_k^{(j)}(s) = (P_immediate^{(j)}(s) - P_replay_k^{(j)}(s)) / P_immediate^{(j)}(s) * 100%
    # FG = 1/(K-1) * Σ_{j=1}^{K-1} 1/|S_j| * Σ_{s∈S_j} 1/(K-j) * Σ_{k=j+1}^K FG_k^{(j)}(s)

    total_fg = 0.0
    num_valid_stages = 0

    print(f"\n{'='*60}")
    print("计算 Forgetting Gain (遗忘度):")

    immediate_all = immediate_performance.get("immediate_all", {})

    for j in range(1, K):  # 遍历所有学习阶段 (除了最后一个)
        samples_in_stage_j = stage_samples[j]
        if not samples_in_stage_j:
            continue

        stage_fg_sum = 0.0
        num_samples_with_replay = 0

        for sample in samples_in_stage_j:
            # 获取该样本的立即测试性能（从主目录test/）
            p_immediate = immediate_all.get(sample)
            if p_immediate is None or p_immediate == 0:
                continue

            # 计算该样本在所有后续 replay 阶段的平均遗忘度
            sample_fg_sum = 0.0
            num_future_replays = 0

            for k in range(j + 1, K + 1):  # 遍历所有后续 replay 阶段
                p_replay = replay_performance[k].get(sample)
                if p_replay is not None:
                    # 计算遗忘度
                    fg = ((p_immediate - p_replay) / p_immediate) * 100.0
                    sample_fg_sum += fg
                    num_future_replays += 1
                    print(f"  阶段{j}学习的样本{sample}, 在阶段{k}测试: immediate={p_immediate:.4f}, replay={p_replay:.4f}, FG={fg:.2f}%")

            if num_future_replays > 0:
                # 对该样本的所有后续 replay 取平均
                avg_sample_fg = sample_fg_sum / num_future_replays
                stage_fg_sum += avg_sample_fg
                num_samples_with_replay += 1

        if num_samples_with_replay > 0:
            # 对该阶段的所有样本取平均
            avg_stage_fg = stage_fg_sum / num_samples_with_replay
            total_fg += avg_stage_fg
            num_valid_stages += 1
            print(f"  阶段{j}平均遗忘度: {avg_stage_fg:.2f}%")

    # 对所有有效阶段取平均
    overall_fg = total_fg / num_valid_stages if num_valid_stages > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"Overall Forgetting Gain: {overall_fg:.2f}%")
    print(f"  (基于 {num_valid_stages} 个学习阶段)")

    # 6. 返回结果
    result = {
        "metric_type": metric_type,
        "num_replay_stages": K,
        "average_success_rate": avg_success_rate,
        "forgetting_gain": overall_fg,
        "num_replay_tests": len(all_replay_scores),
        "num_valid_stages": num_valid_stages
    }

    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python calculate_replay_metrics.py <replay_dir> [metric_type]")
        print("Example: python calculate_replay_metrics.py outputs/replay/mem0/seed66-1 llm_score")
        print("Metric types: llm_score (default), f1_score, bleu_score, average")
        sys.exit(1)

    replay_dir = Path(sys.argv[1])
    metric_type = sys.argv[2] if len(sys.argv) > 2 else "llm_score"

    if not replay_dir.exists():
        print(f"错误: 目录不存在 - {replay_dir}")
        sys.exit(1)

    # 计算指标
    result = calculate_replay_metrics(replay_dir, metric_type)

    # 保存结果
    output_file = replay_dir / f"replay_metrics_{metric_type}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
