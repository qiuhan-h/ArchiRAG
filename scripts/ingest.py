"""把规范文件（或目录）解析、分块并增量入库到 FAISS。

用法：
    python -m scripts.ingest data/raw/消防_建筑设计防火规范_GB 50016-2014.txt
    python -m scripts.ingest data/raw/             # 批量入库整个目录
"""
from __future__ import annotations

import sys
from pathlib import Path

from app.core.parser import load_code_file
from app.core.splitter import BuildingCodeSplitter
from app.core.embedder import using_fallback
from app.core.vector_store import VectorStoreManager

SUPPORTED_SUFFIX = {".txt", ".docx", ".pdf"}


def ingest_one(path: Path, vs: VectorStoreManager) -> int:
    text, meta = load_code_file(path)
    docs = BuildingCodeSplitter(
        meta.profession, meta.code_name, meta.code_no
    ).split(text, source_file=str(path))
    if not docs:
        print(f"  [跳过] 未解析出任何条文: {path.name}")
        return 0
    # 增量更新：先删同规范号旧条文，再写入
    vs.delete_by_code_no(meta.code_no)
    vs.add_documents(docs)
    print(
        f"  [入库] 《{meta.code_name}》{meta.code_no} "
        f"({meta.profession}) → {len(docs)} 条"
    )
    return len(docs)


def main(target: str) -> None:
    p = Path(target)
    if not p.exists():
        raise SystemExit(f"路径不存在: {p}")

    vs = VectorStoreManager()
    print(f"Embedding 模式: {'降级（字符哈希向量）' if using_fallback() else 'bge 真实模型'}")

    files = (
        [f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_SUFFIX]
        if p.is_dir()
        else [p]
    )
    total = sum(ingest_one(f, vs) for f in files)
    print(f"\n完成：处理 {len(files)} 个文件，新增/更新 {total} 条条文，索引总量 {vs.size}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python -m scripts.ingest <规范文件或目录>")
        raise SystemExit(2)
    main(sys.argv[1])
