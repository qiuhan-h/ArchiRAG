"""批量测试脚本：读取 data/test_questions.txt，逐条调用 /api/qa，汇总结果。

用法：
    # 在容器内运行（推荐，避免宿主机中文编码问题）
    docker exec docker-backend-1 python -m scripts.batch_test

    # 指定后端地址（默认 http://localhost:8000）
    python -m scripts.batch_test --api-base http://localhost:8000

    # 只跑某分组（按 test_questions.txt 中的章节标题匹配）
    python -m scripts.batch_test --group 消防
    python -m scripts.batch_test --group 电气 --group 结构

    # 跳过零命中边界测试
    python -m scripts.batch_test --skip-boundary

    # 跳过合规审查（需走 /api/audit 接口，默认跳过）
    python -m scripts.batch_test --with-audit

    # 限制单题超时（秒）
    python -m scripts.batch_test --timeout 60

输出：
    - 控制台逐条打印结果
    - 末尾汇总：通过/失败/缓存命中/平均延迟/降级次数
    - 详细结果写入 data/batch_test_report.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
import urllib.error
from collections import defaultdict
from pathlib import Path

# 默认配置
DEFAULT_API_BASE = "http://localhost:8000"
DEFAULT_QUESTIONS_FILE = "D:\\Trae woek\\ArchiRAG\\data\\ce_shi.txt"
DEFAULT_REPORT_FILE = "data/batch_test_report.json"
DEFAULT_TIMEOUT = 90  # 秒，单题超时


def parse_questions(path: Path) -> list[dict]:
    """解析 test_questions.txt，返回 [{no, group, text}] 列表。

    文件格式：
        ==== 标题 ====         → 分组标题（忽略首行说明）
        1. 问题文本            → 问题
        51. 问题文本（库中无...）→ 边界测试
        # 注释行忽略
        空行忽略
    """
    if not path.exists():
        print(f"[错误] 测试问题文件不存在: {path}", file=sys.stderr)
        return []

    questions: list[dict] = []
    current_group = "未分类"
    in_audit_section = False

    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # 分组标题行：==== xxx ====
            if line.startswith("===="):
                # 提取标题文本
                title = line.strip("= ").strip()
                current_group = title
                in_audit_section = "合规审查" in title
                continue

            # 问题行：数字. 文本
            m = re.match(r"^(\d+)\.\s*(.+)$", line)
            if m:
                no = int(m.group(1))
                text = m.group(2).strip()
                # 去掉行尾说明（括号内）
                clean_text = re.sub(r"[（(].*?[)）]\s*$", "", text).strip()
                questions.append({
                    "no": no,
                    "group": current_group,
                    "text": clean_text,
                    "is_boundary": "零命中" in current_group or "边界" in current_group,
                    "is_audit": in_audit_section,
                })
    return questions


def call_qa(api_base: str, question: str, timeout: int) -> dict:
    """调用 /api/qa，返回响应 dict。"""
    url = f"{api_base.rstrip('/')}/api/qa"
    payload = json.dumps({"question": question}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return {"_error": f"HTTP {e.code}: {body[:200]}"}
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}"}


def call_audit(api_base: str, audit_text: str, timeout: int) -> dict:
    """调用 /api/audit，返回响应 dict。"""
    url = f"{api_base.rstrip('/')}/api/audit"
    payload = json.dumps({"audit_text": audit_text}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return {"_error": f"HTTP {e.code}: {body[:200]}"}
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}"}


def evaluate_qa(q: dict, resp: dict) -> tuple[bool, str]:
    """评估 QA 结果，返回 (是否通过, 说明)。"""
    if "_error" in resp:
        return False, f"请求失败: {resp['_error']}"

    answer = resp.get("answer", "")
    refs = resp.get("references", [])
    is_boundary = q["is_boundary"]

    # 边界测试：应返回"未找到"
    if is_boundary:
        if "未找到" in answer or not refs:
            return True, "边界测试正确返回未找到"
        return False, f"边界测试应返回未找到，实际返回 {len(refs)} 条参考"

    # 正常问题：应有参考条文
    if not refs:
        return False, "未检索到任何参考条文"

    # 检查是否降级
    if resp.get("llm_degraded", True):
        return True, f"降级模式 | 参考 {len(refs)} 条"

    return True, f"参考 {len(refs)} 条 | LLM={resp.get('llm_mode', '?')}"


def evaluate_audit(resp: dict) -> tuple[bool, str]:
    """评估合规审查结果。"""
    if "_error" in resp:
        return False, f"请求失败: {resp['_error']}"

    compliance = resp.get("compliance", "pending")
    violations = resp.get("violations", [])
    hit_mandatory = resp.get("hit_mandatory_count", 0)
    return True, f"compliance={compliance} | 命中强条 {hit_mandatory} | 违规 {len(violations)} 项"


def run_batch(
    api_base: str,
    questions: list[dict],
    timeout: int,
    with_audit: bool,
    skip_boundary: bool,
) -> list[dict]:
    """执行批量测试，返回详细结果列表。"""
    results: list[dict] = []
    total = len(questions)

    for i, q in enumerate(questions, 1):
        # 过滤
        if q["is_audit"] and not with_audit:
            continue
        if q["is_boundary"] and skip_boundary:
            continue

        print(f"\n[{i}/{total}] Q{q['no']} [{q['group']}] {q['text']}")
        t0 = time.time()
        is_audit = q["is_audit"]
        resp = call_audit(api_base, q["text"], timeout) if is_audit else call_qa(api_base, q["text"], timeout)
        elapsed = time.time() - t0

        if is_audit:
            ok, note = evaluate_audit(resp)
        else:
            ok, note = evaluate_qa(q, resp)

        # 提取关键字段
        record = {
            "no": q["no"],
            "group": q["group"],
            "question": q["text"],
            "ok": ok,
            "note": note,
            "elapsed_s": round(elapsed, 2),
            "api_elapsed_ms": resp.get("elapsed_ms", 0) if "_error" not in resp else 0,
            "cache_hit": resp.get("cache_hit", False) if "_error" not in resp else False,
            "llm_degraded": resp.get("llm_degraded", True) if "_error" not in resp else True,
            "reference_count": len(resp.get("references", [])) if "_error" not in resp else 0,
            "answer_preview": (resp.get("answer", "")[:120] + "...") if "_error" not in resp and len(resp.get("answer", "")) > 120 else resp.get("answer", ""),
            "error": resp.get("_error", ""),
        }
        results.append(record)

        status = "PASS" if ok else "FAIL"
        print(f"  → {status} | {note} | 耗时 {elapsed:.1f}s")
        if record["answer_preview"]:
            print(f"  答: {record['answer_preview']}")

    return results


def print_summary(results: list[dict]) -> None:
    """打印汇总报告。"""
    if not results:
        print("\n[汇总] 无可测试结果")
        return

    total = len(results)
    passed = sum(1 for r in results if r["ok"])
    failed = total - passed
    cache_hits = sum(1 for r in results if r["cache_hit"])
    degraded = sum(1 for r in results if r["llm_degraded"])
    avg_ms = sum(r["api_elapsed_ms"] for r in results) / total
    total_wall = sum(r["elapsed_s"] for r in results)

    # 按分组统计
    by_group: dict[str, dict] = defaultdict(lambda: {"total": 0, "pass": 0})
    for r in results:
        g = r["group"]
        by_group[g]["total"] += 1
        if r["ok"]:
            by_group[g]["pass"] += 1

    print("\n" + "=" * 60)
    print("批量测试汇总")
    print("=" * 60)
    print(f"总题数: {total} | 通过: {passed} | 失败: {failed} | 通过率: {passed/total*100:.1f}%")
    print(f"缓存命中: {cache_hits} | LLM 降级: {degraded} | 平均 API 延迟: {avg_ms:.0f}ms")
    print(f"总耗时: {total_wall:.1f}s")
    print("-" * 60)
    print("按分组:")
    for g, s in by_group.items():
        rate = s["pass"] / s["total"] * 100 if s["total"] else 0
        print(f"  {g}: {s['pass']}/{s['total']} ({rate:.0f}%)")
    print("-" * 60)

    if failed:
        print("失败项:")
        for r in results:
            if not r["ok"]:
                print(f"  Q{r['no']} [{r['group']}] {r['question'][:50]}...")
                print(f"    → {r['note']}")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description="ArchiRAG 批量测试")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help=f"后端地址（默认 {DEFAULT_API_BASE}）")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS_FILE, help="测试问题文件路径")
    parser.add_argument("--report", default=DEFAULT_REPORT_FILE, help="报告输出路径")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="单题超时秒数")
    parser.add_argument("--group", action="append", help="只跑指定分组（可多次指定）")
    parser.add_argument("--skip-boundary", action="store_true", help="跳过零命中边界测试")
    parser.add_argument("--with-audit", action="store_true", help="包含合规审查测试（默认跳过）")
    args = parser.parse_args()

    # 解析问题
    questions_path = Path(args.questions)
    if not questions_path.is_absolute():
        # 容器内运行时，工作目录是 /app
        questions_path = Path("/app") / args.questions if Path("/app").exists() else questions_path

    questions = parse_questions(questions_path)
    if not questions:
        print(f"[错误] 未从 {questions_path} 解析到任何问题", file=sys.stderr)
        return 1

    # 按分组过滤
    if args.group:
        questions = [q for q in questions if any(g in q["group"] for g in args.group)]

    # 健康检查
    print(f"后端地址: {args.api_base}")
    try:
        with urllib.request.urlopen(f"{args.api_base.rstrip('/')}/api/health", timeout=10) as resp:
            health = json.loads(resp.read().decode("utf-8"))
            print(f"健康检查: status={health.get('status')} | embedding_fallback={health.get('embedding_fallback')} | llm={health.get('llm_mode')}")
            if health.get("embedding_fallback"):
                print("[警告] Embedding 处于降级模式，语义检索质量会受影响")
    except Exception as e:
        print(f"[错误] 后端不可达: {e}", file=sys.stderr)
        return 1

    print(f"\n待测试题数: {len(questions)}")
    print(f"超时设置: {args.timeout}s/题")
    print("=" * 60)

    # 执行测试
    results = run_batch(args.api_base, questions, args.timeout, args.with_audit, args.skip_boundary)

    # 汇总
    print_summary(results)

    # 写报告
    report_path = Path(args.report)
    if not report_path.is_absolute():
        if Path("/app").exists():
            report_path = Path("/app") / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n详细报告已写入: {report_path}")

    # 失败则返回非零
    failed = sum(1 for r in results if not r["ok"])
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
