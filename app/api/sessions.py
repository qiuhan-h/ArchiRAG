"""会话管理接口 /api/sessions。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.db import crud

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


class SessionCreateRequest(BaseModel):
    title: str = Field("新对话", description="会话标题")


class MessageItem(BaseModel):
    role: str
    content: str
    references: list[dict] = []
    llm_mode: str | None = None
    cache_hit: bool = False
    elapsed_ms: int | None = None
    created_at: str | None = None


@router.get("")
def list_sessions(limit: int = 100):
    """会话列表（最近更新倒序）。"""
    return crud.list_sessions(limit=limit)


@router.post("")
def create_session(req: SessionCreateRequest):
    """新建会话。"""
    sid = crud.create_session(req.title)
    return {"id": sid, "title": req.title[:128]}


@router.get("/{session_id}/messages")
def get_messages(session_id: int):
    """会话内全部消息。"""
    return crud.list_messages(session_id)


@router.delete("/{session_id}")
def delete_session(session_id: int):
    """删除会话。"""
    ok = crud.delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"deleted": True, "id": session_id}
