"""SQLAlchemy 数据模型（严格按方案：三表支撑问答日志与合规审查）。

- CodeDocument：规范条文元数据（与 FAISS 中的 Document 一一对应，
  支撑按规范号/专业/强条统计与展示）；
- QueryLog：每次问答的记录（问题/答案/引用/LLM 模式/耗时/缓存命中），
  支撑查询统计与命中率 KPI；
- AuditRecord：合规审查记录（审查文本/命中强条/合规判定/不合规项）。
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


class CodeDocument(Base):
    """规范条文元数据表。"""

    __tablename__ = "code_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code_no = Column(String(64), index=True, nullable=False)  # 规范号
    code_name = Column(String(128), nullable=False)  # 规范名称
    code_id = Column(String(32), nullable=False)  # 条文编号，如 3.2.1
    profession = Column(String(32), index=True, nullable=False)  # 专业
    is_mandatory = Column(Boolean, default=False, nullable=False)  # 强条
    raw_content = Column(Text, nullable=False)  # 条文原文
    source_file = Column(String(256), nullable=True)  # 来源文件
    file_hash = Column(String(64), nullable=True)  # 文件 hash（增量去重）
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "code_no": self.code_no,
            "code_name": self.code_name,
            "code_id": self.code_id,
            "profession": self.profession,
            "is_mandatory": self.is_mandatory,
            "raw_content": self.raw_content,
            "source_file": self.source_file,
            "file_hash": self.file_hash,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class QueryLog(Base):
    """问答查询日志。"""

    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    references_json = Column(Text, nullable=True)  # JSON 序列化的引用列表
    llm_mode = Column(String(16), nullable=False)  # qwen / fallback
    llm_degraded = Column(Boolean, default=False, nullable=False)
    cache_hit = Column(Boolean, default=False, nullable=False)
    elapsed_ms = Column(Integer, nullable=True)  # 耗时（毫秒）
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True
    )

    def to_dict(self) -> dict:
        import json

        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "references": json.loads(self.references_json) if self.references_json else [],
            "llm_mode": self.llm_mode,
            "llm_degraded": self.llm_degraded,
            "cache_hit": self.cache_hit,
            "elapsed_ms": self.elapsed_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AuditRecord(Base):
    """合规审查记录。"""

    __tablename__ = "audit_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    audit_text = Column(Text, nullable=False)  # 被审查的文本
    profession = Column(String(32), nullable=True)  # 限定专业
    hit_mandatory_count = Column(Integer, default=0, nullable=False)
    compliance = Column(String(16), default="pending", nullable=False)  # pass/fail/pending
    violations_json = Column(Text, nullable=True)  # JSON: 不合规条文列表
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True
    )

    def to_dict(self) -> dict:
        import json

        return {
            "id": self.id,
            "audit_text": self.audit_text,
            "profession": self.profession,
            "hit_mandatory_count": self.hit_mandatory_count,
            "compliance": self.compliance,
            "violations": json.loads(self.violations_json) if self.violations_json else [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ChatSession(Base):
    """问答会话（DeepSeek 风格左侧会话列表）。"""

    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(128), nullable=False)  # 默认取首条问题前 20 字
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True
    )
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class ChatMessage(Base):
    """会话内的单条消息（user / assistant）。"""

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, index=True, nullable=False)
    role = Column(String(16), nullable=False)  # user / assistant
    content = Column(Text, nullable=False)
    references_json = Column(Text, nullable=True)  # assistant 的引用条文
    llm_mode = Column(String(16), nullable=True)
    cache_hit = Column(Boolean, default=False, nullable=False)
    elapsed_ms = Column(Integer, nullable=True)
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True
    )

    def to_dict(self) -> dict:
        import json

        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "references": (
                json.loads(self.references_json) if self.references_json else []
            ),
            "llm_mode": self.llm_mode,
            "cache_hit": self.cache_hit,
            "elapsed_ms": self.elapsed_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
