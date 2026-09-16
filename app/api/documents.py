"""文档管理接口 /api/documents。"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.core.parser import load_code_file
from app.core.splitter import BuildingCodeSplitter
from app.core.vector_store import VectorStoreManager
from app.db import crud
from app.schemas.models import (
    DeleteRequest,
    DeleteResponse,
    DocumentItem,
    DocumentListResponse,
    IngestRequest,
    IngestResponse,
)

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.get("", response_model=DocumentListResponse)
def list_documents(
    profession: str | None = None,
    code_no: str | None = None,
    only_mandatory: bool = False,
    limit: int = 200,
) -> DocumentListResponse:
    """列出规范条文元数据。"""
    rows = crud.get_code_docs(
        profession=profession,
        code_no=code_no,
        only_mandatory=only_mandatory,
        limit=limit,
    )
    return DocumentListResponse(
        total=len(rows),
        items=[DocumentItem(**r) for r in rows],
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest_document(req: IngestRequest) -> IngestResponse:
    """入库规范文件（解析→分块→FAISS+DB）。"""
    p = Path(req.file_path)
    if not p.is_absolute():
        p = settings.watch_dir_abs / p
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在: {p}")

    # 解析 + 分块
    text, meta = load_code_file(p)
    docs = BuildingCodeSplitter(
        meta.profession, meta.code_name, meta.code_no
    ).split(text, source_file=str(p))

    if not docs:
        raise HTTPException(status_code=422, detail="未解析出任何条文")

    # FAISS 增量入库
    vs = VectorStoreManager()
    vs.delete_by_code_no(meta.code_no)
    vs.add_documents(docs)

    # DB 入库
    file_hash = crud.compute_file_hash(str(p), text)
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
        source_file=str(p),
        file_hash=file_hash,
    )

    return IngestResponse(
        code_no=meta.code_no,
        code_name=meta.code_name,
        profession=meta.profession,
        ingested=len(docs),
    )


@router.delete("", response_model=DeleteResponse)
def delete_documents(req: DeleteRequest) -> DeleteResponse:
    """删除某本规范的全部条文（FAISS + DB）。"""
    vs = VectorStoreManager()
    faiss_deleted = vs.delete_by_code_no(req.code_no)
    db_deleted = crud.delete_code_docs(req.code_no)
    return DeleteResponse(
        deleted=max(faiss_deleted, db_deleted),
        code_no=req.code_no,
    )
