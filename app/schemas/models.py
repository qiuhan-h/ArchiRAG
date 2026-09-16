"""Pydantic 请求/响应模型（对齐文档接口规范）。"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# 问答
# ------------------------------------------------------------------
class QARequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    profession: str | None = Field(None, description="限定专业（消防/电气/结构…）")
    k: int | None = Field(None, ge=1, le=50, description="返回条数")
    only_mandatory: bool = Field(False, description="只检索强制性条文")


class ReferenceItem(BaseModel):
    code_name: str
    code_no: str
    code_id: str
    is_mandatory: bool
    content: str


class QAResponse(BaseModel):
    answer: str
    references: list[ReferenceItem] = []
    llm_mode: str = "fallback"
    llm_degraded: bool = True
    cache_hit: bool = False
    elapsed_ms: int = 0


class StreamQARequest(QARequest):
    session_id: int | None = Field(None, description="会话 ID；为空则新建会话")


# ------------------------------------------------------------------
# 合规审查
# ------------------------------------------------------------------
class AuditRequest(BaseModel):
    audit_text: str = Field(..., min_length=1, description="被审查的文本")
    profession: str | None = Field(None, description="限定专业")


class AuditResponse(BaseModel):
    compliance: str  # pass / fail / pending
    hit_mandatory_count: int
    violations: list[dict[str, Any]] = []
    references: list[ReferenceItem] = []
    audit_id: int | None = None


# ------------------------------------------------------------------
# 统计
# ------------------------------------------------------------------
class StatsResponse(BaseModel):
    by_profession: list[dict[str, Any]] = []
    query_stats: dict[str, Any] = {}


# ------------------------------------------------------------------
# 文档管理
# ------------------------------------------------------------------
class DocumentItem(BaseModel):
    code_no: str
    code_name: str
    code_id: str
    profession: str
    is_mandatory: bool
    raw_content: str
    source_file: str | None = None


class DocumentListResponse(BaseModel):
    total: int
    items: list[DocumentItem]


class IngestRequest(BaseModel):
    file_path: str = Field(..., description="规范文件路径（data/raw 下）")


class IngestResponse(BaseModel):
    code_no: str
    code_name: str
    profession: str
    ingested: int


class DeleteRequest(BaseModel):
    code_no: str


class DeleteResponse(BaseModel):
    deleted: int
    code_no: str


# ------------------------------------------------------------------
# 通用
# ------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = "ok"
    db_type: str = ""
    cache_type: str = ""
    llm_mode: str = ""
    embedding_fallback: bool = True
