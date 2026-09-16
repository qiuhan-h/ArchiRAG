"""合规审查接口 /api/audit。"""
from __future__ import annotations

from fastapi import APIRouter

from app.core.audit_service import AuditService
from app.db import crud
from app.schemas.models import AuditRequest, AuditResponse, ReferenceItem

router = APIRouter(prefix="/api/audit", tags=["audit"])

_audit_service: AuditService | None = None


def _get_service() -> AuditService:
    global _audit_service
    if _audit_service is None:
        _audit_service = AuditService()
    return _audit_service


@router.post("", response_model=AuditResponse)
def audit(req: AuditRequest) -> AuditResponse:
    """合规审查：分句检索强条 → 合规判定。"""
    result = _get_service().audit(req.audit_text, profession=req.profession)

    audit_id = crud.create_audit(
        audit_text=req.audit_text,
        profession=req.profession,
        hit_mandatory_count=result.hit_mandatory_count,
        compliance=result.compliance,
        violations=result.violations,
    )

    return AuditResponse(
        compliance=result.compliance,
        hit_mandatory_count=result.hit_mandatory_count,
        violations=result.violations,
        references=[ReferenceItem(**r) for r in result.references],
        audit_id=audit_id,
    )
