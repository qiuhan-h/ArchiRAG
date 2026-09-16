"""CRUD 操作（按方案：问答日志 + 规范条文 upsert + 专业统计）。

全部通过 session.py 的 get_session_ctx 上下文管理器操作，
支持 MySQL / SQLite 两种后端。
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import case, func

from app.db.models import AuditRecord, ChatMessage, ChatSession, CodeDocument, QueryLog
from app.db.session import get_session_ctx

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# QueryLog
# ------------------------------------------------------------------
def log_query(
    question: str,
    answer: str,
    references: list[dict[str, Any]],
    llm_mode: str,
    llm_degraded: bool,
    cache_hit: bool,
    elapsed_ms: int | None = None,
) -> int:
    """记录一次问答查询，返回日志 ID。"""
    with get_session_ctx() as session:
        log = QueryLog(
            question=question,
            answer=answer,
            references_json=json.dumps(references, ensure_ascii=False),
            llm_mode=llm_mode,
            llm_degraded=llm_degraded,
            cache_hit=cache_hit,
            elapsed_ms=elapsed_ms,
        )
        session.add(log)
        session.flush()
        return log.id


def recent_queries(limit: int = 50) -> list[dict]:
    """最近 N 条查询日志。"""
    with get_session_ctx() as session:
        rows = (
            session.query(QueryLog)
            .order_by(QueryLog.created_at.desc())
            .limit(limit)
            .all()
        )
        return [r.to_dict() for r in rows]


# ------------------------------------------------------------------
# CodeDocument
# ------------------------------------------------------------------
def compute_file_hash(source_file: str, content: str) -> str:
    """文件内容 hash（用于增量同步去重）。"""
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def upsert_code_docs(
    code_no: str,
    code_name: str,
    profession: str,
    docs: list[dict[str, Any]],
    source_file: str = "",
    file_hash: str = "",
) -> int:
    """规范条文批量 upsert（先删旧再写新，返回写入条数）。

    docs 中每项需含 code_id / raw_content / is_mandatory。
    """
    with get_session_ctx() as session:
        # 先删同规范号旧条文
        session.query(CodeDocument).filter(
            CodeDocument.code_no == code_no
        ).delete(synchronize_session=False)

        for d in docs:
            session.add(
                CodeDocument(
                    code_no=code_no,
                    code_name=code_name,
                    profession=profession,
                    code_id=d["code_id"],
                    raw_content=d["raw_content"],
                    is_mandatory=d.get("is_mandatory", False),
                    source_file=source_file,
                    file_hash=file_hash,
                )
            )
        session.flush()
        return len(docs)


def get_code_docs(
    profession: str | None = None,
    code_no: str | None = None,
    only_mandatory: bool = False,
    limit: int = 200,
) -> list[dict]:
    """查询规范条文元数据。"""
    with get_session_ctx() as session:
        q = session.query(CodeDocument)
        if profession:
            q = q.filter(CodeDocument.profession == profession)
        if code_no:
            q = q.filter(CodeDocument.code_no == code_no)
        if only_mandatory:
            q = q.filter(CodeDocument.is_mandatory.is_(True))
        rows = q.order_by(CodeDocument.code_id).limit(limit).all()
        return [r.to_dict() for r in rows]


def delete_code_docs(code_no: str) -> int:
    """删除某本规范的全部条文，返回删除条数。"""
    with get_session_ctx() as session:
        return (
            session.query(CodeDocument)
            .filter(CodeDocument.code_no == code_no)
            .delete(synchronize_session=False)
        )


# ------------------------------------------------------------------
# AuditRecord
# ------------------------------------------------------------------
def create_audit(
    audit_text: str,
    profession: str | None,
    hit_mandatory_count: int,
    compliance: str,
    violations: list[dict[str, Any]],
) -> int:
    """记录一次合规审查，返回审查记录 ID。"""
    with get_session_ctx() as session:
        rec = AuditRecord(
            audit_text=audit_text,
            profession=profession,
            hit_mandatory_count=hit_mandatory_count,
            compliance=compliance,
            violations_json=json.dumps(violations, ensure_ascii=False),
        )
        session.add(rec)
        session.flush()
        return rec.id


def recent_audits(limit: int = 50) -> list[dict]:
    """最近 N 条审查记录。"""
    with get_session_ctx() as session:
        rows = (
            session.query(AuditRecord)
            .order_by(AuditRecord.created_at.desc())
            .limit(limit)
            .all()
        )
        return [r.to_dict() for r in rows]


# ------------------------------------------------------------------
# 会话（ChatSession / ChatMessage）
# ------------------------------------------------------------------
def create_session(title: str) -> int:
    """创建会话，返回会话 ID。"""
    with get_session_ctx() as session:
        s = ChatSession(title=title[:128])
        session.add(s)
        session.flush()
        return s.id


def list_sessions(limit: int = 100) -> list[dict]:
    """会话列表（按最近更新倒序）。"""
    with get_session_ctx() as session:
        rows = (
            session.query(ChatSession)
            .order_by(ChatSession.updated_at.desc())
            .limit(limit)
            .all()
        )
        return [r.to_dict() for r in rows]


def delete_session(session_id: int) -> bool:
    """删除会话及其全部消息。"""
    with get_session_ctx() as session:
        session.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).delete(synchronize_session=False)
        deleted = (
            session.query(ChatSession)
            .filter(ChatSession.id == session_id)
            .delete(synchronize_session=False)
        )
        return deleted > 0


def add_message(
    session_id: int,
    role: str,
    content: str,
    references: list[dict[str, Any]] | None = None,
    llm_mode: str | None = None,
    cache_hit: bool = False,
    elapsed_ms: int | None = None,
) -> int:
    """追加一条消息；user 消息会刷新会话 updated_at（列表置顶）。"""
    with get_session_ctx() as session:
        msg = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            references_json=(
                json.dumps(references, ensure_ascii=False) if references else None
            ),
            llm_mode=llm_mode,
            cache_hit=cache_hit,
            elapsed_ms=elapsed_ms,
        )
        session.add(msg)
        session.flush()
        return msg.id


def list_messages(session_id: int) -> list[dict]:
    """会话内全部消息（时间正序）。"""
    with get_session_ctx() as session:
        rows = (
            session.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id.asc())
            .all()
        )
        return [r.to_dict() for r in rows]


def touch_session(session_id: int) -> None:
    """显式刷新会话 updated_at。"""
    with get_session_ctx() as session:
        s = session.get(ChatSession, session_id)
        if s is not None:
            s.updated_at = datetime.now(timezone.utc)


# ------------------------------------------------------------------
# 统计
# ------------------------------------------------------------------
def stats_by_profession() -> list[dict]:
    """按专业统计条文数和强条数（用 case 兼容 SQLite/MySQL）。"""
    with get_session_ctx() as session:
        rows = (
            session.query(
                CodeDocument.profession,
                func.count(CodeDocument.id).label("total"),
                func.sum(
                    case(
                        (CodeDocument.is_mandatory.is_(True), 1),
                        else_=0,
                    )
                ).label("mandatory"),
            )
            .group_by(CodeDocument.profession)
            .all()
        )
        return [
            {
                "profession": r.profession,
                "total": r.total,
                "mandatory": int(r.mandatory or 0),
            }
            for r in rows
        ]


def query_stats() -> dict:
    """查询统计：总数/缓存命中数/平均耗时/降级次数。"""
    with get_session_ctx() as session:
        total = session.query(func.count(QueryLog.id)).scalar() or 0
        cache_hits = (
            session.query(func.count(QueryLog.id))
            .filter(QueryLog.cache_hit.is_(True))
            .scalar()
            or 0
        )
        degraded = (
            session.query(func.count(QueryLog.id))
            .filter(QueryLog.llm_degraded.is_(True))
            .scalar()
            or 0
        )
        avg_ms = session.query(func.avg(QueryLog.elapsed_ms)).scalar()
        return {
            "total_queries": total,
            "cache_hits": cache_hits,
            "cache_hit_rate": round(cache_hits / total, 4) if total else 0.0,
            "avg_latency_ms": round(float(avg_ms), 1) if avg_ms else 0.0,
            "degraded_count": degraded,
        }
