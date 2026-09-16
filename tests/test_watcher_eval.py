"""M6 测试：文件监控增量同步 + hash 去重 + 评估脚本。"""
import os
import time
from pathlib import Path

import pytest


# ------------------------------------------------------------------
# 文件监控
# ------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _reset_env(monkeypatch, temp_index):
    import app.db.session as sess
    import app.core.cache as cache_mod
    from app.config import settings

    _TEST_DB = os.path.join(
        os.path.dirname(__file__), ".tmp_indexes", "test_watcher.db"
    )
    if os.path.exists(_TEST_DB):
        os.remove(_TEST_DB)
    monkeypatch.setattr(settings, "SQLITE_PATH", _TEST_DB)
    monkeypatch.setattr(settings, "REDIS_URL", "redis://localhost:1/0")
    sess._engine = None
    sess._SessionLocal = None
    sess._db_type = ""
    cache_mod._cache = None
    cache_mod._cache_type = ""
    yield
    if sess._engine is not None:
        sess._engine.dispose()
    sess._engine = None
    sess._SessionLocal = None
    sess._db_type = ""
    if os.path.exists(_TEST_DB):
        try:
            os.remove(_TEST_DB)
        except PermissionError:
            pass


def test_ingest_file_hashes_and_dedup(temp_index, monkeypatch):
    """相同内容文件不重复入库。"""
    from app.watcher.file_monitor import ingest_file, _get_stored_hashes
    from app.core.vector_store import VectorStoreManager

    # 准备测试文件
    watch_dir = temp_index.parent / "watch_src"
    watch_dir.mkdir(exist_ok=True)
    f = watch_dir / "test_消防_GB-001-2026.txt"
    f.write_text("1 防火\n1.1.1 普通条文，描述防火要求。\n", encoding="utf-8")

    from app.config import settings
    monkeypatch.setattr(settings, "WATCH_DIR", str(watch_dir))

    stored = {}
    # 第一次入库
    assert ingest_file(f, stored) is True
    assert len(stored) == 1

    # FAISS 有条文
    vs = VectorStoreManager()
    assert vs.size > 0

    # 第二次相同内容 → 跳过
    assert ingest_file(f, stored) is False


def test_full_scan(temp_index, monkeypatch):
    """全量扫描目录入库所有文件。"""
    from app.watcher.file_monitor import full_scan

    watch_dir = temp_index.parent / "watch_scan"
    watch_dir.mkdir(exist_ok=True)
    (watch_dir / "a_消防_GB-A-2026.txt").write_text(
        "1 防火\n1.1.1 防火要求内容。\n", encoding="utf-8"
    )
    (watch_dir / "b_电气_GB-B-2026.txt").write_text(
        "1 电气\n1.1.1 电气要求内容。\n", encoding="utf-8"
    )
    # 非规范文件应被忽略
    (watch_dir / "readme.md").write_text("not a code", encoding="utf-8")

    from app.config import settings
    monkeypatch.setattr(settings, "WATCH_DIR", str(watch_dir))

    count = full_scan()
    assert count == 2  # 2 个 txt 文件入库


def test_full_scan_empty_dir(temp_index, monkeypatch):
    from app.watcher.file_monitor import full_scan

    empty_dir = temp_index.parent / "empty_watch"
    empty_dir.mkdir(exist_ok=True)
    from app.config import settings
    monkeypatch.setattr(settings, "WATCH_DIR", str(empty_dir))
    assert full_scan() == 0


# ------------------------------------------------------------------
# 评估脚本
# ------------------------------------------------------------------
@pytest.fixture
def code_for_eval(temp_index):
    """入库演示规范供评估。"""
    from app.core.parser import load_code_file
    from app.core.splitter import BuildingCodeSplitter
    from app.core.vector_store import VectorStoreManager
    from app.config import settings

    demo_path = settings.watch_dir_abs.parent / "examples" / "消防_演示规范_DEMO-001-2026.txt"
    if demo_path.exists():
        text, meta = load_code_file(demo_path)
        docs = BuildingCodeSplitter(
            meta.profession, meta.code_name, meta.code_no
        ).split(text, "demo")
        VectorStoreManager().add_documents(docs)
    else:
        # fallback：直接用内联文本
        code = """1 总则
1.0.1 为指导建筑防火设计、预防建筑火灾、减少火灾危害，制定本演示规范。
1.0.2 本演示规范规定的防火要求应与国家现行有关标准协调使用。
2 建筑分类与耐火等级
2.1.1 【强制性条文】厂房和仓库的耐火等级分为一、二、三、四级，并应符合相应构件的燃烧性能和耐火极限要求。
2.1.2 【强制性条文】重要公共建筑的耐火等级**不应低于二级**，且主要承重构件必须采用不燃材料。
3 防火间距
3.1.1 【强制性条文】民用建筑之间的防火间距应根据建筑耐火等级和建筑高度确定，相邻两座建筑的防火间距不得小于规定限值。
3.1.2 【强制性条文】甲类厂房与重要公共建筑的防火间距严禁小于 50m。
"""
        docs = BuildingCodeSplitter("消防", "演示规范", "DEMO-001-2026").split(code, "eval.txt")
        VectorStoreManager().add_documents(docs)
    return temp_index


def test_eval_demo_set(code_for_eval):
    from scripts.evaluate import DEMO_EVAL_SET, evaluate
    from app.core.qa_service import QAService

    qa = QAService()
    metrics = evaluate(qa, DEMO_EVAL_SET, k=8)

    assert metrics["total"] == 7
    assert metrics["correct"] == 7  # 全部正确
    assert metrics["accuracy"] == 1.0
    assert metrics["avg_latency_ms"] < 2000  # KPI: <2s


def test_eval_report_format(code_for_eval):
    from scripts.evaluate import DEMO_EVAL_SET, evaluate, print_report
    from app.core.qa_service import QAService

    qa = QAService()
    metrics = evaluate(qa, DEMO_EVAL_SET)
    # print_report 不抛异常即可
    print_report(metrics)
    assert "accuracy" in metrics
    assert "results" in metrics
