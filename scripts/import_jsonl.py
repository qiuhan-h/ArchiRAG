"""把已结构化的 JSONL 条文数据导入 FAISS + 数据库（幂等，可重复执行）。

数据每行一条 JSON：
    {"id":"NT-100","profession":"暖通","code_name":"测试用编撰条文",
     "code_id":"NT-100","content":"...","is_mandatory":false,"source":"编撰"}

字段映射：
    profession/code_name/code_id/content/is_mandatory 直接映射；
    code_no 缺失时按 code_name + profession 推导（同名跨专业时细化，
    保证一个 code_no 唯一对应一本"规范批次"，重复导入先删后写）；
    source / id 作为附加元数据存入 FAISS（DB 不强制要求）。
    source / id 作为附加元数据存入 FAISS（DB 不强制要求）。

用法：
    python -m scripts.import_jsonl data/seed/shu_ju.jsonl
    python -m scripts.import_jsonl /app/data/seed/shu_ju.jsonl   # 容器内
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from langchain_core.documents import Document

from app.core.embedder import using_fallback
from app.core.vector_store import VectorStoreManager
from app.db import crud

REQUIRED_FIELDS = ("profession", "code_name", "code_id", "content")


def load_rows(path: Path) -> tuple[list[dict], list[tuple[int, str]]]:
    """逐行读取 JSONL，返回 (有效行, 坏行[(行号, 原因)])。UTF-8 BOM 兼容。"""
    rows: list[dict] = []
    bad: list[tuple[int, str]] = []
    with open(path, encoding="utf-8-sig") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as e:
                bad.append((lineno, f"JSON 解析失败: {e}"))
                continue
            if not isinstance(row, dict):
                bad.append((lineno, "该行不是 JSON 对象"))
                continue
            missing = [k for k in REQUIRED_FIELDS if not str(row.get(k, "")).strip()]
            if missing:
                bad.append((lineno, f"缺少/为空必需字段: {','.join(missing)}"))
                continue
            rows.append(row)
    return rows, bad


def derive_code_nos(rows: list[dict]) -> dict[int, str]:
    """为每行确定 code_no（行内优先；缺失时单专业用 code_name，跨专业细化）。"""
    name_to_profs: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if not r.get("code_no"):
            name_to_profs[r["code_name"]].add(r["profession"])

    mapping: dict[int, str] = {}
    for i, r in enumerate(rows):
        if r.get("code_no"):
            mapping[i] = str(r["code_no"]).strip()
        elif len(name_to_profs[r["code_name"]]) == 1:
            mapping[i] = r["code_name"]
        else:
            mapping[i] = f"{r['code_name']}·{r['profession']}"
    return mapping


def to_document(row: dict, code_no: str, source_file: str) -> Document:
    content = str(row["content"]).strip()
    metadata = {
        "profession": str(row["profession"]).strip(),
        "code_name": str(row["code_name"]).strip(),
        "code_no": code_no,
        "code_id": str(row["code_id"]).strip(),
        "is_mandatory": bool(row.get("is_mandatory", False)),
        "source_file": source_file,
        "raw_content": content,
        "explanation": str(row.get("explanation", "")).strip(),
        # 附加信息（FAISS 保存，检索链路不依赖）
        "source": str(row.get("source", "")).strip(),
        "source_id": str(row.get("id", "")).strip(),
    }
    return Document(page_content=content, metadata=metadata)


def main() -> int:
    parser = argparse.ArgumentParser(description="JSONL 条文数据导入")
    parser.add_argument("jsonl_path", help="shu_ju.jsonl 路径")
    args = parser.parse_args()

    path = Path(args.jsonl_path)
    if not path.exists():
        print(f"[错误] 文件不存在: {path}", file=sys.stderr)
        return 2

    rows, bad = load_rows(path)
    print(f"读取 {path.name}：有效 {len(rows)} 行，坏行 {len(bad)} 行")
    for lineno, reason in bad[:20]:
        print(f"  [坏行 {lineno}] {reason}")
    if len(bad) > 20:
        print(f"  …另有 {len(bad) - 20} 条坏行未显示")
    if not rows:
        print("[终止] 没有可导入的有效数据", file=sys.stderr)
        return 1

    code_no_map = derive_code_nos(rows)

    # 按 code_no 分组（同组 code_name/profession 必一致）
    groups: dict[str, list[dict]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[code_no_map[i]].append(r)

    file_hash = hashlib.md5(path.read_bytes()).hexdigest()
    source_file = path.name
    vs = VectorStoreManager()
    print(
        f"Embedding 模式: {'降级（字符哈希向量）' if using_fallback() else 'bge 真实模型'}"
    )
    print(f"待导入规范批次: {len(groups)} 个")

    total = 0
    mandatory_total = 0
    for code_no, group in sorted(groups.items()):
        code_name = str(group[0]["code_name"]).strip()
        profession = str(group[0]["profession"]).strip()
        docs = [to_document(r, code_no, source_file) for r in group]

        # FAISS + DB 双写，先删旧批次 → 幂等
        vs.delete_by_code_no(code_no)
        vs.add_documents(docs)
        crud.upsert_code_docs(
            code_no=code_no,
            code_name=code_name,
            profession=profession,
            docs=[
                {
                    "code_id": d.metadata["code_id"],
                    "raw_content": d.metadata["raw_content"],
                    "is_mandatory": d.metadata["is_mandatory"],
                }
                for d in docs
            ],
            source_file=source_file,
            file_hash=file_hash,
        )
        n_man = sum(d.metadata["is_mandatory"] for d in docs)
        mandatory_total += n_man
        total += len(docs)
        print(
            f"  [入库] 《{code_name}》{code_no}（{profession}）"
            f"→ {len(docs)} 条，其中强条 {n_man} 条"
        )

    prof_dist = Counter(str(r["profession"]).strip() for r in rows)
    print(f"\n完成：{total} 条条文（强条 {mandatory_total} 条），坏行 {len(bad)} 条")
    print("专业分布: " + "，".join(f"{k}={v}" for k, v in prof_dist.items()))
    print(f"FAISS 索引总量: {vs.size}")
    return 0  # 有坏行但有效数据已入库，仍算成功（坏行行号已打印）


if __name__ == "__main__":
    raise SystemExit(main())
