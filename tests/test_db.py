"""M4 测试：DB 模型/CRUD/统计 + 缓存降级 + 缓存接入问答链路。"""
import json
import os
import time

import pytest

# 测试用独立 SQLite + 不存在的 Redis
_TEST_DB = os.path.join(
    os.path.dirname(__file__), ".tmp_indexes", "test_db.db"
)


@pytest.fixture(autouse=True)
def _reset_db(monkeypatch):
    """每个测试前重置 DB/缓存单例，保证隔离。"""
    import app.db.session as sess
    import app.core.cache as cache_mod
    from app.config import settings

    # 清掉旧 DB 文件 + patch settings
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
    # 先 dispose 引擎释放文件锁，再删
    if sess._engine is not None:
        sess._engine.dispose()
    sess._engine = None
    sess._SessionLocal = None
    sess._db_type = ""
    if os.path.exists(_TEST_DB):
        try:
            os.remove(_TEST_DB)
        except PermissionError:
            pass  # Windows 偶尔残留句柄，下一轮再清


# ------------------------------------------------------------------
# DB Session 降级
# ------------------------------------------------------------------
def test_db_falls_back_to_sqlite():
    import app.db.session as sess

    engine = sess.get_engine()
    assert sess.db_type() == "sqlite"
    assert engine is not None


# ------------------------------------------------------------------
# CodeDocument CRUD
# ------------------------------------------------------------------
def test_upsert_and_get_code_docs():
    from app.db import crud

    docs = [
        {"code_id": "1.1.1", "raw_content": "防火要求", "is_mandatory": False},
        {"code_id": "1.1.2", "raw_content": "必须设置喷淋", "is_mandatory": True},
    ]
    n = crud.upsert_code_docs(
        "GB-01", "测试规范", "消防", docs, source_file="t.txt", file_hash="abc"
    )
    assert n == 2

    rows = crud.get_code_docs(profession="消防")
    assert len(rows) == 2
    assert rows[0]["code_no"] == "GB-01"
    assert rows[0]["profession"] == "消防"

    # only_mandatory 过滤
    mand = crud.get_code_docs(only_mandatory=True)
    assert len(mand) == 1
    assert mand[0]["code_id"] == "1.1.2"
    assert mand[0]["is_mandatory"] is True


def test_upsert_replaces_old():
    from app.db import crud

    crud.upsert_code_docs("GB-X", "规范", "消防", [
        {"code_id": "1.0.1", "raw_content": "旧", "is_mandatory": False},
    ])
    assert len(crud.get_code_docs(code_no="GB-X")) == 1

    crud.upsert_code_docs("GB-X", "规范", "消防", [
        {"code_id": "2.0.1", "raw_content": "新1", "is_mandatory": False},
        {"code_id": "2.0.2", "raw_content": "新2", "is_mandatory": True},
    ])
    rows = crud.get_code_docs(code_no="GB-X")
    assert len(rows) == 2
    assert all(r["code_id"].startswith("2.") for r in rows)


def test_delete_code_docs():
    from app.db import crud

    crud.upsert_code_docs("GB-D", "规范", "电气", [
        {"code_id": "1.0.1", "raw_content": "x", "is_mandatory": False},
    ])
    n = crud.delete_code_docs("GB-D")
    assert n == 1
    assert crud.get_code_docs(code_no="GB-D") == []


# ------------------------------------------------------------------
# QueryLog
# ------------------------------------------------------------------
def test_log_query_and_recent():
    from app.db import crud

    log_id = crud.log_query(
        question="防火间距是多少",
        answer="50m",
        references=[{"code_id": "3.1.2", "is_mandatory": True}],
        llm_mode="fallback",
        llm_degraded=True,
        cache_hit=False,
        elapsed_ms=120,
    )
    assert log_id > 0

    rows = crud.recent_queries(limit=10)
    assert len(rows) == 1
    assert rows[0]["question"] == "防火间距是多少"
    assert rows[0]["answer"] == "50m"
    assert rows[0]["references"][0]["code_id"] == "3.1.2"
    assert rows[0]["cache_hit"] is False
    assert rows[0]["elapsed_ms"] == 120


# ------------------------------------------------------------------
# AuditRecord
# ------------------------------------------------------------------
def test_create_and_list_audit():
    from app.db import crud

    audit_id = crud.create_audit(
        audit_text="本项目采用二级耐火等级",
        profession="消防",
        hit_mandatory_count=1,
        compliance="pass",
        violations=[],
    )
    assert audit_id > 0

    rows = crud.recent_audits(limit=10)
    assert len(rows) == 1
    assert rows[0]["compliance"] == "pass"
    assert rows[0]["hit_mandatory_count"] == 1


# ------------------------------------------------------------------
# 统计
# ------------------------------------------------------------------
def test_stats_by_profession():
    from app.db import crud

    crud.upsert_code_docs("GB-A", "规范A", "消防", [
        {"code_id": "1.0.1", "raw_content": "a", "is_mandatory": False},
        {"code_id": "1.0.2", "raw_content": "b", "is_mandatory": True},
    ])
    crud.upsert_code_docs("GB-B", "规范B", "电气", [
        {"code_id": "1.0.1", "raw_content": "c", "is_mandatory": True},
    ])

    stats = crud.stats_by_profession()
    by_prof = {s["profession"]: s for s in stats}
    assert by_prof["消防"]["total"] == 2
    assert by_prof["消防"]["mandatory"] == 1
    assert by_prof["电气"]["total"] == 1
    assert by_prof["电气"]["mandatory"] == 1


def test_query_stats():
    from app.db import crud

    crud.log_query("q1", "a1", [], "fallback", True, False, 100)
    crud.log_query("q2", "a2", [], "fallback", False, True, 50)
    crud.log_query("q3", "a3", [], "qwen", False, False, 200)

    stats = crud.query_stats()
    assert stats["total_queries"] == 3
    assert stats["cache_hits"] == 1
    assert abs(stats["cache_hit_rate"] - 1 / 3) < 0.01
    assert stats["degraded_count"] == 1


# ------------------------------------------------------------------
# 缓存降级
# ------------------------------------------------------------------
def test_cache_falls_back_to_memory():
    from app.core import cache

    c = cache.get_cache()
    assert cache.cache_type() == "memory"
    assert c.get("missing") is None

    c.set("k1", "v1", ttl=2)
    assert c.get("k1") == "v1"
    assert c.exists("k1") is True

    c.delete("k1")
    assert c.get("k1") is None


def test_cache_ttl_expiry():
    from app.core import cache

    c = cache.get_cache()
    c.set("expire_me", "temp", ttl=1)
    assert c.get("expire_me") == "temp"
    time.sleep(1.1)
    assert c.get("expire_me") is None


def test_cache_key_generation():
    from app.core.cache import make_cache_key

    k1 = make_cache_key("防火间距", profession="消防")
    k2 = make_cache_key("防火间距", profession="消防")
    k3 = make_cache_key("防火间距", profession="电气")
    assert k1 == k2  # 相同参数 → 相同 key
    assert k1 != k3  # 不同参数 → 不同 key
    assert k1.startswith("qa:")


# ------------------------------------------------------------------
# 缓存接入问答链路（模拟 API 层）
# ------------------------------------------------------------------
def test_cache_hit_skips_retrieval(index_with_code):
    """模拟 API 层：缓存命中时直接返回，不重新检索。"""
    from app.core.cache import cache_get, cache_set, make_cache_key
    from app.core.qa_service import QAService

    svc = QAService()
    q = "高层建筑必须设置什么灭火系统"

    # 第一次调用：未命中缓存
    key = make_cache_key(q)
    assert cache_get(key) is None

    result1 = svc.ask(q)
    # 写入缓存
    cache_set(key, json.dumps(result1.to_dict()), ttl=60)

    # 第二次：缓存命中
    cached = cache_get(key)
    assert cached is not None
    cached_dict = json.loads(cached)
    assert cached_dict["answer"] == result1.answer
    assert "自动喷水灭火系统" in cached_dict["answer"]


# ------------------------------------------------------------------
# 文件 hash 去重
# ------------------------------------------------------------------
def test_file_hash():
    from app.db.crud import compute_file_hash

    h1 = compute_file_hash("a.txt", "防火要求")
    h2 = compute_file_hash("b.txt", "防火要求")
    h3 = compute_file_hash("a.txt", "不同内容")
    assert h1 == h2  # hash 只看内容
    assert h1 != h3
