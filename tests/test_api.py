"""M5 API 端到端测试（FastAPI TestClient）。"""
import os

import pytest
from fastapi.testclient import TestClient

# 测试环境配置
_TEST_DB = os.path.join(
    os.path.dirname(__file__), ".tmp_indexes", "test_api.db"
)


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch, temp_index):
    """每个测试前重置 DB/缓存/FAISS 单例。"""
    import app.db.session as sess
    import app.core.cache as cache_mod
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

    # 重置 API 服务单例
    import app.api.qa as qa_mod
    import app.api.audit as audit_mod
    qa_mod._qa_service = None
    audit_mod._audit_service = None

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


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


@pytest.fixture
def code_in_index(temp_index):
    """入库测试规范到 FAISS。"""
    from app.core.splitter import BuildingCodeSplitter
    from app.core.vector_store import VectorStoreManager

    code = """1 防火
1.1.1 普通条文，描述一般防火要求。
1.1.2 【强制性条文】高层建筑**必须**设置自动喷水灭火系统。
"""
    docs = BuildingCodeSplitter("消防", "演示防火规范", "GB-API").split(
        code, source_file="api.txt"
    )
    VectorStoreManager().add_documents(docs)
    return temp_index


# ------------------------------------------------------------------
# 健康检查
# ------------------------------------------------------------------
def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["db_type"] == "sqlite"
    assert data["cache_type"] == "memory"
    assert data["llm_mode"] == "fallback"


# ------------------------------------------------------------------
# 问答
# ------------------------------------------------------------------
def test_qa_hit(client, code_in_index):
    resp = client.post("/api/qa", json={
        "question": "高层建筑必须设置什么灭火系统",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "自动喷水灭火系统" in data["answer"]
    assert data["references"]
    assert data["references"][0]["code_id"] == "1.1.2"
    assert data["llm_mode"] == "fallback"
    assert data["cache_hit"] is False
    assert data["elapsed_ms"] >= 0


def test_qa_cache_hit(client, code_in_index):
    """第二次相同问题应命中缓存。"""
    payload = {"question": "高层建筑必须设置什么灭火系统"}
    r1 = client.post("/api/qa", json=payload)
    assert r1.json()["cache_hit"] is False

    r2 = client.post("/api/qa", json=payload)
    assert r2.json()["cache_hit"] is True
    assert r2.json()["answer"] == r1.json()["answer"]


def test_qa_not_found(client, code_in_index):
    resp = client.post("/api/qa", json={
        "question": "幕墙四性试验检测费用由谁承担",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "未找到" in data["answer"]
    assert data["references"] == []


def test_qa_empty_question_rejected(client):
    resp = client.post("/api/qa", json={"question": ""})
    assert resp.status_code == 422  # Pydantic validation


def test_qa_only_mandatory(client, code_in_index):
    resp = client.post("/api/qa", json={
        "question": "防火",
        "only_mandatory": True,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert all(r["is_mandatory"] for r in data["references"])


# ------------------------------------------------------------------
# 合规审查
# ------------------------------------------------------------------
def test_audit_pass_no_mandatory(client, code_in_index):
    """审查文本与强条无关联 → pass。"""
    resp = client.post("/api/audit", json={
        "audit_text": "本项目位于城市边缘地带，交通便利。",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["compliance"] == "pass"
    assert data["hit_mandatory_count"] == 0
    assert data["audit_id"] is not None


def test_audit_fail_with_mandatory(client, code_in_index):
    """审查文本命中强条 → fail（降级模式标记需人工复核）。"""
    resp = client.post("/api/audit", json={
        "audit_text": "高层建筑未设置自动喷水灭火系统。",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["compliance"] == "fail"
    assert data["hit_mandatory_count"] > 0
    assert data["violations"]
    assert "code_id" in data["violations"][0]


def test_audit_empty_text_rejected(client):
    resp = client.post("/api/audit", json={"audit_text": ""})
    assert resp.status_code == 422


# ------------------------------------------------------------------
# 统计
# ------------------------------------------------------------------
def test_stats_empty(client):
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "by_profession" in data
    assert "query_stats" in data


def test_stats_after_queries(client, code_in_index):
    # 先问一轮
    client.post("/api/qa", json={"question": "高层建筑必须设置什么灭火系统"})
    client.post("/api/qa", json={"question": "高层建筑必须设置什么灭火系统"})

    resp = client.get("/api/stats")
    data = resp.json()
    assert data["query_stats"]["total_queries"] == 2
    assert data["query_stats"]["cache_hits"] == 1


# ------------------------------------------------------------------
# 文档管理
# ------------------------------------------------------------------
def test_documents_list_empty(client):
    resp = client.get("/api/documents")
    assert resp.status_code == 200
    assert resp.json()["total"] == 0


def test_documents_list_after_ingest(client, code_in_index):
    # 入库到 DB
    from app.db import crud
    crud.upsert_code_docs("GB-API", "演示防火规范", "消防", [
        {"code_id": "1.1.1", "raw_content": "普通", "is_mandatory": False},
        {"code_id": "1.1.2", "raw_content": "强条", "is_mandatory": True},
    ])

    resp = client.get("/api/documents")
    data = resp.json()
    assert data["total"] == 2

    # 按强条过滤
    resp2 = client.get("/api/documents?only_mandatory=true")
    assert resp2.json()["total"] == 1


def test_documents_delete(client, code_in_index):
    from app.db import crud
    crud.upsert_code_docs("GB-DEL", "待删规范", "消防", [
        {"code_id": "1.0.1", "raw_content": "x", "is_mandatory": False},
    ])
    assert len(crud.get_code_docs(code_no="GB-DEL")) == 1

    resp = client.request("DELETE", "/api/documents", json={"code_no": "GB-DEL"})
    assert resp.status_code == 200
    assert resp.json()["deleted"] >= 1
