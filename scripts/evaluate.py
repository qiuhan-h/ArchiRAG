"""评估脚本 —— 支撑 91% 准确率 KPI。

评估方式：
1. 准备评估集（questions.jsonl），每行一条 {question, expected_code_ids, profession}；
2. 对每条 question 调 QAService.ask()，取返回的 references；
3. 计算：
   - recall@k：期望条文是否出现在返回结果中；
   - precision：返回结果中有多少是期望条文；
   - answer_quality：降级模式只检查答案是否含期望关键词；
4. 汇总输出准确率（达标线 91%）。

用法：
    python -m scripts.evaluate --eval-file tests/eval_questions.jsonl
    python -m scripts.evaluate --demo  # 用演示规范跑内置评估集
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from app.core.qa_service import QAService


def load_eval_set(path: str) -> list[dict]:
    """加载评估集（JSONL）。"""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


DEMO_EVAL_SET = [
    {
        "question": "甲类厂房与重要公共建筑的防火间距",
        "expected_code_ids": ["3.1.2"],
        "profession": None,
        "keywords": ["50", "50m", "五十"],
    },
    {
        "question": "重要公共建筑的耐火等级要求",
        "expected_code_ids": ["2.1.2"],
        "profession": None,
        "keywords": ["二级", "耐火"],
    },
    {
        "question": "第2.1.2条",
        "expected_code_ids": ["2.1.2"],
        "profession": None,
        "keywords": [],
    },
    {
        "question": "民用建筑防火间距",
        "expected_code_ids": ["3.1.1"],
        "profession": None,
        "keywords": ["防火间距"],
    },
    {
        "question": "厂房和仓库的耐火等级",
        "expected_code_ids": ["2.1.1"],
        "profession": None,
        "keywords": ["一级", "二级", "三级", "四级"],
    },
    {
        "question": "幕墙四性试验检测费用",
        "expected_code_ids": [],
        "profession": None,
        "keywords": ["未找到"],
    },
    {
        "question": "基坑支护锚杆长度计算",
        "expected_code_ids": [],
        "profession": None,
        "keywords": ["未找到"],
    },
]


def evaluate(qa: QAService, eval_set: list[dict], k: int = 8) -> dict:
    """运行评估，返回指标。"""
    results = []
    for item in eval_set:
        t0 = time.time()
        r = qa.ask(
            question=item["question"],
            profession=item.get("profession"),
            k=k,
        )
        elapsed_ms = int((time.time() - t0) * 1000)

        returned_ids = [ref["code_id"] for ref in r.references]
        expected_ids = item.get("expected_code_ids", [])

        # recall: 期望条文是否全部出现在返回中
        if expected_ids:
            hit = sum(1 for eid in expected_ids if eid in returned_ids)
            recall = hit / len(expected_ids)
        else:
            # 期望"未找到"
            recall = 1.0 if not returned_ids else 0.0

        # answer keyword check
        keywords = item.get("keywords", [])
        if keywords:
            kw_hit = any(kw in r.answer for kw in keywords)
        else:
            kw_hit = True

        results.append({
            "question": item["question"],
            "expected": expected_ids,
            "returned": returned_ids,
            "recall": recall,
            "keyword_hit": kw_hit,
            "elapsed_ms": elapsed_ms,
            "answer_snippet": r.answer[:80],
        })

    # 汇总
    total = len(results)
    avg_recall = sum(r["recall"] for r in results) / total if total else 0
    avg_kw = sum(r["keyword_hit"] for r in results) / total if total else 0
    # 综合准确率 = recall 和 keyword 都通过的占比
    correct = sum(
        1 for r in results if r["recall"] >= 1.0 and r["keyword_hit"]
    )
    accuracy = correct / total if total else 0
    avg_ms = sum(r["elapsed_ms"] for r in results) / total if total else 0

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "avg_recall": round(avg_recall, 4),
        "keyword_pass_rate": round(avg_kw, 4),
        "avg_latency_ms": round(avg_ms, 1),
        "results": results,
    }


def print_report(metrics: dict) -> None:
    """打印评估报告。"""
    print("=" * 60)
    print("ArchiRAG 检索评估报告")
    print("=" * 60)
    print(f"评估条数:     {metrics['total']}")
    print(f"完全正确:     {metrics['correct']}")
    print(f"准确率:       {metrics['accuracy']:.1%}  (KPI 目标: 91%)")
    print(f"平均召回:     {metrics['avg_recall']:.1%}")
    print(f"关键词命中:   {metrics['keyword_pass_rate']:.1%}")
    print(f"平均耗时:     {metrics['avg_latency_ms']}ms  (KPI 目标: <2000ms)")
    print("-" * 60)
    for r in metrics["results"]:
        status = "PASS" if r["recall"] >= 1.0 and r["keyword_hit"] else "FAIL"
        print(f"[{status}] {r['question'][:30]}")
        print(f"  期望: {r['expected']}  返回: {r['returned'][:5]}")
        print(f"  recall={r['recall']:.0%} kw={r['keyword_hit']} {r['elapsed_ms']}ms")
    print("=" * 60)
    target = 0.91
    if metrics["accuracy"] >= target:
        print(f"达标 ({metrics['accuracy']:.1%} >= {target:.0%})")
    else:
        print(f"未达标 ({metrics['accuracy']:.1%} < {target:.0%})")


def main():
    parser = argparse.ArgumentParser(description="ArchiRAG 评估脚本")
    parser.add_argument(
        "--eval-file", type=str, help="评估集 JSONL 路径"
    )
    parser.add_argument(
        "--demo", action="store_true", help="用演示规范内置评估集"
    )
    parser.add_argument("--k", type=int, default=8, help="Top-K")
    args = parser.parse_args()

    if args.demo:
        eval_set = DEMO_EVAL_SET
    elif args.eval_file:
        eval_set = load_eval_set(args.eval_file)
    else:
        print("请指定 --demo 或 --eval-file")
        sys.exit(2)

    qa = QAService()
    metrics = evaluate(qa, eval_set, k=args.k)
    print_report(metrics)


if __name__ == "__main__":
    main()
