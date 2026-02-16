#!/usr/bin/env python3
"""
计算 Repair 模式的评估指标
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


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
        # result.result.metrics 中包含指标
        metrics = result.get("result", {}).get("result", {}).get("metrics", {})
        return metrics.get(metric_type, 0.0)
    except Exception as e:
        print(f"Warning: Failed to extract metric: {e}")
        return None


def calculate_repair_metrics(repair_dir: Path, metric_type: str = "llm_score") -> Dict:
    """
    计算 Repair 模式的评估指标

    Args:
        repair_dir: repair 目录路径 (e.g., outputs/repair/mem0/seed66-1)
        metric_type: 指标类型 (llm_score, f1_score, bleu_score)

    Returns:
        包含以下指标的字典：
        - repair_gain_full: 所有样本的修复增益
        - repair_gain_standard: 反转样本的修复增益
        - num_repair_stages: repair 阶段数量
        - avg_wrong_full: 所有样本在错误学习后的平均性能
        - avg_right_full: 所有样本在修复后的平均性能
        - avg_wrong_standard: 反转样本在错误学习后的平均性能
        - avg_right_standard: 反转样本在修复后的平均性能
    """
    repair_dir = Path(repair_dir)

    # 1. 找到所有 repair 阶段目录
    repair_stages = sorted([d for d in repair_dir.iterdir() if d.is_dir() and d.name.startswith("repair")],
                          key=lambda x: int(x.name.replace("repair", "")))

    K = len(repair_stages)
    print(f"\n{'='*60}")
    print(f"处理目录: {repair_dir}")
    print(f"找到 {K} 个 repair 阶段")

    # 2. 收集所有阶段的性能数据
    all_wrong_full_scores = []
    all_right_full_scores = []
    all_wrong_standard_scores = []
    all_right_standard_scores = []

    for stage_idx, stage_dir in enumerate(repair_stages, start=1):
        print(f"\n处理阶段 {stage_idx}: {stage_dir.name}")

        # 读取四个测试目录的结果
        for test_type in ["wrongJudgeTestFull", "rightJudgeTestFull",
                         "wrongJudgeTestStandard", "rightJudgeTestStandard"]:
            test_dir = stage_dir / test_type

            if not test_dir.exists():
                print(f"  Warning: {test_type} 目录不存在")
                continue

            stage_scores = []

            # 遍历所有任务目录
            for task_dir in test_dir.iterdir():
                if task_dir.is_dir():
                    task_name = task_dir.name

                    # 读取所有样本结果
                    for sample_file in task_dir.glob("*.json"):
                        result = load_sample_result(sample_file)
                        if result:
                            perf = get_performance(result, metric_type)
                            if perf is not None:
                                stage_scores.append(perf)

            # 存储到对应的列表
            if test_type == "wrongJudgeTestFull":
                all_wrong_full_scores.extend(stage_scores)
                print(f"  {test_type}: {len(stage_scores)} 个样本, 平均 {sum(stage_scores)/len(stage_scores):.4f}")
            elif test_type == "rightJudgeTestFull":
                all_right_full_scores.extend(stage_scores)
                print(f"  {test_type}: {len(stage_scores)} 个样本, 平均 {sum(stage_scores)/len(stage_scores):.4f}")
            elif test_type == "wrongJudgeTestStandard":
                all_wrong_standard_scores.extend(stage_scores)
                print(f"  {test_type}: {len(stage_scores)} 个样本, 平均 {sum(stage_scores)/len(stage_scores):.4f}")
            elif test_type == "rightJudgeTestStandard":
                all_right_standard_scores.extend(stage_scores)
                print(f"  {test_type}: {len(stage_scores)} 个样本, 平均 {sum(stage_scores)/len(stage_scores):.4f}")

    # 3. 计算平均性能
    avg_wrong_full = sum(all_wrong_full_scores) / len(all_wrong_full_scores) if all_wrong_full_scores else 0.0
    avg_right_full = sum(all_right_full_scores) / len(all_right_full_scores) if all_right_full_scores else 0.0
    avg_wrong_standard = sum(all_wrong_standard_scores) / len(all_wrong_standard_scores) if all_wrong_standard_scores else 0.0
    avg_right_standard = sum(all_right_standard_scores) / len(all_right_standard_scores) if all_right_standard_scores else 0.0

    # 4. 计算 Repair Gain
    repair_gain_full = avg_right_full - avg_wrong_full
    repair_gain_standard = avg_right_standard - avg_wrong_standard

    print(f"\n{'='*60}")
    print(f"修复性能统计:")
    print(f"\n【Full (所有样本)】")
    print(f"  错误学习后正确率_full:  {avg_wrong_full:.4f} ({avg_wrong_full*100:.2f}%)")
    print(f"  修复学习后正确率_full:  {avg_right_full:.4f} ({avg_right_full*100:.2f}%)")
    print(f"  修复增益率_full:        {repair_gain_full:.4f} ({repair_gain_full*100:.2f}%)")

    print(f"\n【Standard (反转样本)】")
    print(f"  错误学习后正确率_standard:  {avg_wrong_standard:.4f} ({avg_wrong_standard*100:.2f}%)")
    print(f"  修复学习后正确率_standard:  {avg_right_standard:.4f} ({avg_right_standard*100:.2f}%)")
    print(f"  修复增益率_standard:        {repair_gain_standard:.4f} ({repair_gain_standard*100:.2f}%)")

    print(f"\n样本数统计:")
    print(f"  wrongJudgeTestFull: {len(all_wrong_full_scores)} 个样本")
    print(f"  rightJudgeTestFull: {len(all_right_full_scores)} 个样本")
    print(f"  wrongJudgeTestStandard: {len(all_wrong_standard_scores)} 个样本")
    print(f"  rightJudgeTestStandard: {len(all_right_standard_scores)} 个样本")

    # 5. 返回结果
    result = {
        "metric_type": metric_type,
        "num_repair_stages": K,
        "repair_gain_full": repair_gain_full,
        "repair_gain_standard": repair_gain_standard,
        "avg_wrong_full": avg_wrong_full,
        "avg_right_full": avg_right_full,
        "avg_wrong_standard": avg_wrong_standard,
        "avg_right_standard": avg_right_standard,
        "num_samples_full": len(all_wrong_full_scores),
        "num_samples_standard": len(all_wrong_standard_scores)
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="计算 Repair 模式的评估指标")
    parser.add_argument("repair_dir", type=str, help="Repair 目录路径")
    parser.add_argument("--metric", type=str, default="llm_score",
                       choices=["llm_score", "f1_score", "bleu_score"],
                       help="指标类型 (默认: llm_score)")
    parser.add_argument("--output", type=str, default=None,
                       help="输出JSON文件路径 (可选)")

    args = parser.parse_args()

    # 计算指标
    result = calculate_repair_metrics(args.repair_dir, args.metric)

    # 保存结果
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.repair_dir) / f"repair_metrics_{args.metric}.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
