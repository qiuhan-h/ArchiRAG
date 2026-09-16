"""统计接口 /api/stats。"""
from __future__ import annotations

from fastapi import APIRouter

from app.db import crud
from app.schemas.models import StatsResponse

router = APIRouter(prefix="/api/stats", tags=["stats"])


@router.get("", response_model=StatsResponse)
def get_stats() -> StatsResponse:
    """专业条文统计 + 查询统计。"""
    return StatsResponse(
        by_profession=crud.stats_by_profession(),
        query_stats=crud.query_stats(),
    )
