"""文件监控 + 增量同步（独立进程运行，避免多 worker 重复入库）。

关键设计：
1. 独立进程：通过 `python -m app.watcher.file_monitor` 启动，
   不与 FastAPI worker 混在同一进程内，避免 uvicorn --workers >1
   时每个 worker 各起一个 watcher 导致重复入库；
2. hash 去重：文件内容 hash 与 DB 中已存 hash 比对，相同则跳过；
3. 事件处理：on_created / on_modified → 触发增量入库；
4. 启动时全量扫描一次 watch_dir，补齐遗漏文件。
"""
from __future__ import annotations

import hashlib
import logging
import os
import sys
import time
from pathlib import Path

from app.config import settings
from app.core.parser import load_code_file
from app.core.splitter import BuildingCodeSplitter
from app.core.vector_store import VectorStoreManager
from app.db import crud
from app.db.session import get_engine

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIX = {".txt", ".docx", ".pdf"}


def _file_content_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _get_stored_hashes() -> dict[str, str]:
    """获取 DB 中已入库的文件 hash（source_file -> file_hash）。"""
    with crud.get_session_ctx() as session:
        from sqlalchemy import distinct, select
        from app.db.models import CodeDocument

        rows = (
            session.query(
                CodeDocument.source_file,
                CodeDocument.file_hash,
            )
            .distinct()
            .all()
        )
        return {r[0]: r[1] for r in rows if r[0]}


def ingest_file(path: Path, stored_hashes: dict[str, str]) -> bool:
    """增量入库单个文件，返回是否实际入库。"""
    str_path = str(path)
    if path.suffix.lower() not in SUPPORTED_SUFFIX:
        return False

    try:
        text, meta = load_code_file(path)
    except Exception as e:
        logger.error("解析失败 %s: %s", path.name, e)
        return False

    content_hash = _file_content_hash(text)
    old_hash = stored_hashes.get(str_path)
    if old_hash == content_hash:
        logger.debug("跳过（hash 未变）: %s", path.name)
        return False

    docs = BuildingCodeSplitter(
        meta.profession, meta.code_name, meta.code_no
    ).split(text, source_file=str_path)
    if not docs:
        logger.warning("未解析出条文: %s", path.name)
        return False

    # FAISS 增量
    vs = VectorStoreManager()
    vs.delete_by_code_no(meta.code_no)
    vs.add_documents(docs)

    # DB 入库
    doc_dicts = [
        {
            "code_id": d.metadata["code_id"],
            "raw_content": d.metadata.get("raw_content", ""),
            "is_mandatory": d.metadata.get("is_mandatory", False),
        }
        for d in docs
    ]
    crud.upsert_code_docs(
        code_no=meta.code_no,
        code_name=meta.code_name,
        profession=meta.profession,
        docs=doc_dicts,
        source_file=str_path,
        file_hash=content_hash,
    )
    stored_hashes[str_path] = content_hash
    logger.info(
        "增量入库: 《%s》%s (%s) → %d 条",
        meta.code_name,
        meta.code_no,
        meta.profession,
        len(docs),
    )
    return True


def full_scan() -> int:
    """全量扫描 watch_dir，补齐遗漏文件，返回入库文件数。"""
    watch_dir = settings.watch_dir_abs
    if not watch_dir.exists():
        logger.warning("监控目录不存在: %s", watch_dir)
        return 0

    stored_hashes = _get_stored_hashes()
    count = 0
    for f in sorted(watch_dir.rglob("*")):
        if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIX:
            if ingest_file(f, stored_hashes):
                count += 1
    logger.info("全量扫描完成: 入库 %d 个文件", count)
    return count


def start_monitor() -> None:
    """启动文件监控（阻塞，独立进程入口）。"""
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    watch_dir = settings.watch_dir_abs
    logger.info("启动文件监控: %s", watch_dir)

    # 1. 启动时全量扫描
    stored_hashes = _get_stored_hashes()
    full_scan()

    # 2. 增量监控
    class Handler(FileSystemEventHandler):
        def _maybe_ingest(self, path_str: str) -> None:
            p = Path(path_str)
            if not p.exists() or p.suffix.lower() not in SUPPORTED_SUFFIX:
                return
            # 防抖：文件可能还在写入
            time.sleep(1)
            ingest_file(p, stored_hashes)

        def on_created(self, event) -> None:
            if not event.is_directory:
                logger.info("检测到新文件: %s", event.src_path)
                self._maybe_ingest(event.src_path)

        def on_modified(self, event) -> None:
            if not event.is_directory:
                self._maybe_ingest(event.src_path)

    observer = Observer()
    handler = Handler()
    observer.schedule(handler, str(watch_dir), recursive=True)
    observer.start()

    logger.info("文件监控已启动，按 Ctrl+C 停止")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("文件监控停止中…")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # 确保 DB 引擎已初始化
    get_engine()
    start_monitor()
