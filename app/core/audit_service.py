"""合规审查服务（文档中后端实现缺失，此处补全）。

审查逻辑：
1. 对输入文本分句（按句号/分号/换行切分）；
2. 每句去检索强制性条文（only_mandatory=True）；
3. 命中强条的句子标记为"需关注"；
4. 若任何命中强条与审查文本存在冲突（如审查文本说"耐火等级三级"
   而强条要求"不应低于二级"），标记为 violation；
5. 无任何强条命中 → pass；有命中但无冲突 → pass with warnings；
   有冲突 → fail。

简化实现（降级模式）：因无真实 LLM 做语义冲突判定，
violation 列表只记录命中的强条供人工复核。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

from app.core.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)

# 句子切分：中文句号、分号、换行
_SENT_SPLIT = re.compile(r"[。；;\n]+")


def _split_sentences(text: str) -> list[str]:
    """把审查文本切成句子。"""
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


@dataclass
class AuditResult:
    compliance: str  # pass / fail / pending
    hit_mandatory_count: int
    violations: list[dict[str, Any]] = field(default_factory=list)
    references: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "compliance": self.compliance,
            "hit_mandatory_count": self.hit_mandatory_count,
            "violations": self.violations,
            "references": self.references,
        }


class AuditService:
    """合规审查服务。"""

    # 简化的冲突关键词（真实场景需 LLM 语义判定）
    _CONFLICT_KEYWORDS = ["不低于", "不应低于", "严禁", "必须", "不得"]

    def __init__(self, retriever: HybridRetriever | None = None) -> None:
        self.retriever = retriever or HybridRetriever()

    def audit(self, audit_text: str, profession: str | None = None) -> AuditResult:
        sentences = _split_sentences(audit_text)
        if not sentences:
            return AuditResult(compliance="pending", hit_mandatory_count=0)

        all_refs: list[dict[str, Any]] = []
        seen_code_ids: set[str] = set()
        violations: list[dict[str, Any]] = []

        for sent in sentences:
            # 只检索强制性条文
            docs = self.retriever.search(
                sent, profession=profession, k=5, only_mandatory=True
            )
            for d in docs:
                cid = d.metadata.get("code_id", "")
                if cid in seen_code_ids:
                    continue
                seen_code_ids.add(cid)

                ref = {
                    "code_name": d.metadata["code_name"],
                    "code_no": d.metadata["code_no"],
                    "code_id": cid,
                    "is_mandatory": True,
                    "content": d.metadata.get("raw_content", ""),
                }
                all_refs.append(ref)

                # 简化冲突检测：命中强条即标记为"需关注"
                # （真实 LLM 模式下应做语义冲突判定）
                violations.append({
                    "sentence": sent,
                    "code_id": cid,
                    "code_name": d.metadata["code_name"],
                    "mandatory_clause": d.metadata.get("raw_content", ""),
                    "note": "命中强制性条文，请人工复核是否合规",
                })

        mandatory_count = len(all_refs)
        if mandatory_count == 0:
            compliance = "pass"
        else:
            # 有命中强条，简化为"需关注"（降级模式不做语义冲突判定）
            compliance = "fail"

        return AuditResult(
            compliance=compliance,
            hit_mandatory_count=mandatory_count,
            violations=violations,
            references=all_refs,
        )
