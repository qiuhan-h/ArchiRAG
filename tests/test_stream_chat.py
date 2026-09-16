"""M6-前端改造：SSE 流式问答 + 会话管理 测试。"""
import json
import os

import pytest
from fastapi.testclient import TestClient

_TEST_DB = os.path.join(
    os.path.dirname(__file__), ".tmp_indexes", "test_stream.db"
)


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch, temp_index):
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

    import app.api.qa as qa_mod
    qa_mod._qa_service = None

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
    from app.core.splitter import BuildingCodeSplitter
    from app.core.vector_store import VectorStoreManager

    code = """1 防火
1.1.1 普通条文，描述一般防火要求。
1.1.2 【强制性条文】高层建筑**必须**设置自动喷水灭火系统。
"""
    docs = BuildingCodeSplitter("消防", "演示防火规范", "GB-STR").split(
        code, source_file="stream.txt"
    )
    VectorStoreManager().add_documents(docs)
    return temp_index


def parse_sse(text: str) -> list[dict]:
    """解析 SSE 响应文本为事件列表。"""
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


# ------------------------------------------------------------------
# QAService.ask_stream 事件协议
# ------------------------------------------------------------------
def test_ask_stream_events(code_in_index):
    from app.core.qa_service import QAService

    svc = QAService()
    events = list(svc.ask_stream("高层建筑必须设置什么灭火系统"))

    types = [e["type"] for e in events]
    assert types[0] == "meta"
    assert types[-1] == "done"
    assert types.count("meta") == 1

    meta = events[0]
    assert meta["references"]
    assert meta["references"][0]["code_id"] == "1.1.2"
    assert meta["preset_answer"] is None
    assert meta["llm_mode"] == "fallback"

    # token 拼接即完整答案
    tokens = [e["delta"] for e in events if e["type"] == "token"]
    assert len(tokens) > 1  # 降级模式分块流式
    full = "".join(tokens)
    assert "自动喷水灭火系统" in full


def test_ask_stream_not_found_preset(code_in_index):
    from app.core.qa_service import QAService

    events = list(QAService().ask_stream("幕墙四性试验检测费用由谁承担"))
    meta = events[0]
    assert meta["references"] == []
    assert "未找到" in meta["preset_answer"]
    # preset 路径没有 token 事件
    assert not [e for e in events if e["type"] == "token"]
    assert events[-1]["type"] == "done"


# ------------------------------------------------------------------
# SSE 端点 + 会话持久化
# ------------------------------------------------------------------
def test_stream_endpoint_creates_session(client, code_in_index):
    resp = client.post("/api/qa/stream", json={
        "question": "高层建筑必须设置什么灭火系统",
    })
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    events = parse_sse(resp.text)
    meta = events[0]
    assert meta["type"] == "meta"
    assert meta["cache_hit"] is False
    assert meta["session_id"] is not None
    assert events[-1]["type"] == "done"

    full = "".join(e["delta"] for e in events if e["type"] == "token")
    assert "自动喷水灭火系统" in full

    # 会话与消息已持久化
    from app.db import crud
    sid = meta["session_id"]
    msgs = crud.list_messages(sid)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "高层建筑必须设置什么灭火系统"
    assert "自动喷水灭火系统" in msgs[1]["content"]
    assert msgs[1]["references"][0]["code_id"] == "1.1.2"

    sessions = crud.list_sessions()
    assert len(sessions) == 1
    assert "高层" in sessions[0]["title"]


def test_stream_reuse_session(client, code_in_index):
    """带 session_id 的第二轮问答进入同一会话。"""
    r1 = parse_sse(client.post("/api/qa/stream", json={
        "question": "高层建筑必须设置什么灭火系统",
    }).text)
    sid = r1[0]["session_id"]

    r2 = parse_sse(client.post("/api/qa/stream", json={
        "question": "一般防火要求",
        "session_id": sid,
    }).text)
    assert r2[0]["session_id"] == sid

    from app.db import crud
    msgs = crud.list_messages(sid)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert len(crud.list_sessions()) == 1


def test_stream_cache_hit_preset(client, code_in_index):
    r1 = parse_sse(client.post("/api/qa/stream", json={
        "question": "高层建筑必须设置什么灭火系统",
    }).text)
    assert r1[0]["cache_hit"] is False

    r2 = parse_sse(client.post("/api/qa/stream", json={
        "question": "高层建筑必须设置什么灭火系统",
    }).text)
    assert r2[0]["cache_hit"] is True
    assert "自动喷水灭火系统" in r2[0]["preset_answer"]


def test_stream_not_found_persisted(client, code_in_index):
    events = parse_sse(client.post("/api/qa/stream", json={
        "question": "幕墙四性试验检测费用由谁承担",
    }).text)
    sid = events[0]["session_id"]
    from app.db import crud
    msgs = crud.list_messages(sid)
    assert "未找到" in msgs[1]["content"]


# ------------------------------------------------------------------
# 会话 REST 接口
# ------------------------------------------------------------------
def test_sessions_api_lifecycle(client, code_in_index):
    # 无会话
    assert client.get("/api/sessions").json() == []

    # 流式问答产生会话
    parse_sse(client.post("/api/qa/stream", json={
        "question": "高层建筑必须设置什么灭火系统",
    }).text)

    listing = client.get("/api/sessions").json()
    assert len(listing) == 1
    sid = listing[0]["id"]

    msgs = client.get(f"/api/sessions/{sid}/messages").json()
    assert len(msgs) == 2

    # 删除
    assert client.delete(f"/api/sessions/{sid}").status_code == 200
    assert client.get("/api/sessions").json() == []
    assert client.get(f"/api/sessions/{sid}/messages").json() == []

    # 再删 404
    assert client.delete(f"/api/sessions/{sid}").status_code == 404


def test_create_session_explicit(client):
    resp = client.post("/api/sessions", json={"title": "手动会话"})
    assert resp.status_code == 200
    sid = resp.json()["id"]
    listing = client.get("/api/sessions").json()
    assert listing[0]["title"] == "手动会话"
    assert client.delete(f"/api/sessions/{sid}").status_code == 200
