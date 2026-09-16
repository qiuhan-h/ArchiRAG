"""JSONL 导入器测试：坏行容错、code_no 推导、FAISS+DB 双写、幂等。"""
import json
import os
import sys

import pytest

_TEST_DB = os.path.join(
    os.path.dirname(__file__), ".tmp_indexes", "test_import.db"
)


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch, temp_index):
    import app.db.session as sess
    import app.core.cache as cache_mod
    import app.core.vector_store as vs_mod
    from app.config import settings

    if os.path.exists(_TEST_DB):
        os.remove(_TEST_DB)
    monkeypatch.setattr(settings, "SQLITE_PATH", _TEST_DB)
    monkeypatch.setattr(settings, "REDIS_URL", "redis://localhost:1/0")

    sess._engine = None
    sess._SessionLocal = None
    sess._db_type = ""
    cache_mod._cache = None
    cache_mod._cache_type = ""
    vs_mod.VectorStoreManager._instance = None

    yield

    if sess._engine is not None:
        sess._engine.dispose()
    sess._engine = None
    sess._SessionLocal = None
    sess._db_type = ""
    vs_mod.VectorStoreManager._instance = None
    if os.path.exists(_TEST_DB):
        try:
            os.remove(_TEST_DB)
        except PermissionError:
            pass


def _write_jsonl(path, rows: list[dict], inject_bad: bool = False) -> None:
    lines = [json.dumps(r, ensure_ascii=False) for r in rows]
    if inject_bad:
        lines.insert(1, "{这不是合法 json")
        lines.insert(2, json.dumps({"profession": "缺字段"}, ensure_ascii=False))
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_import(monkeypatch, path) -> int:
    monkeypatch.setattr(sys, "argv", ["import_jsonl", str(path)])
    from scripts.import_jsonl import main
    return main()


def test_import_basic_and_idempotent(monkeypatch, temp_index):
    from app.core.vector_store import VectorStoreManager
    from app.db import crud
    from scripts.import_jsonl import load_rows

    rows = [
        {"id": "XF-001", "profession": "消防", "code_name": "消防规范A",
         "code_id": "1.0.1", "content": "高层建筑必须设置自动喷水灭火系统。",
         "is_mandatory": True},
        {"id": "XF-002", "profession": "消防", "code_name": "消防规范A",
         "code_id": "1.0.2", "content": "普通条文内容。", "is_mandatory": False},
        {"id": "DQ-001", "profession": "电气", "code_name": "消防规范A",
         "code_id": "2.0.1", "content": "电气专业的条文内容。",
         "is_mandatory": False},
        {"id": "GB-001", "profession": "结构", "code_name": "带真实规范号",
         "code_no": "GB 9999-2020", "code_id": "3.0.1",
         "content": "真实规范号优先使用。", "is_mandatory": False},
    ]
    f = temp_index / "seed.jsonl"
    _write_jsonl(f, rows, inject_bad=True)

    valid, bad = load_rows(f)
    assert len(valid) == 4
    assert len(bad) == 2  # 非法 JSON + 缺必需字段

    assert _run_import(monkeypatch, f) == 0

    # 4 条全部入库 FAISS
    assert VectorStoreManager().size == 4

    # 跨专业同名 → 两个批次；真实 code_no 保留
    docs = VectorStoreManager().all_documents()
    code_nos = {d.metadata["code_no"] for d in docs}
    assert "消防规范A·消防" in code_nos
    assert "消防规范A·电气" in code_nos
    assert "GB 9999-2020" in code_nos

    # DB 三个批次、专业正确
    assert len(crud.get_code_docs(code_no="消防规范A·消防")) == 2
    assert len(crud.get_code_docs(code_no="消防规范A·电气")) == 1
    assert len(crud.get_code_docs(code_no="GB 9999-2020")) == 1

    # 幂等：再导一次数量不变（先删后写）
    assert _run_import(monkeypatch, f) == 0
    assert VectorStoreManager().size == 4


def test_single_profession_uses_code_name(monkeypatch, temp_index):
    """同名只跨一个专业时 code_no 直接用 code_name。"""
    from app.core.vector_store import VectorStoreManager
    from app.db import crud

    rows = [
        {"id": "A1", "profession": "暖通", "code_name": "暖通规范B",
         "code_id": "1.1.1", "content": "暖通条文一。", "is_mandatory": False},
        {"id": "A2", "profession": "暖通", "code_name": "暖通规范B",
         "code_id": "1.1.2", "content": "暖通条文二。", "is_mandatory": False},
    ]
    f = temp_index / "b.jsonl"
    _write_jsonl(f, rows)
    assert _run_import(monkeypatch, f) == 0
    assert VectorStoreManager().size == 2
    assert len(crud.get_code_docs(code_no="暖通规范B")) == 2
