"""ArchiRAG FastAPI 主应用。

启动：
    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import (
    audit_router,
    documents_router,
    qa_router,
    sessions_router,
    stats_router,
)
from app.core.embedder import using_fallback
from app.core import llm
from app.db.session import db_type, get_engine
from app.core.cache import cache_type, get_cache
from app.schemas.models import HealthResponse

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动：初始化 DB 引擎 + 缓存（惰性触发）
    get_engine()
    get_cache()
    logger.info(
        "ArchiRAG 启动完成 | DB=%s | Cache=%s | LLM=%s | Embedding 降级=%s",
        db_type(),
        cache_type(),
        llm.llm_mode(),
        using_fallback(),
    )
    yield


app = FastAPI(
    title="ArchiRAG 建筑规范智能问答与合规审查系统",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(qa_router)
app.include_router(sessions_router)
app.include_router(audit_router)
app.include_router(stats_router)
app.include_router(documents_router)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """健康检查：当前各组件运行模式。"""
    return HealthResponse(
        db_type=db_type(),
        cache_type=cache_type(),
        llm_mode=llm.llm_mode(),
        embedding_fallback=using_fallback(),
    )
