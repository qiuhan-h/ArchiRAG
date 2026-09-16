"""问答主流程（核心业务链路）。

检索（HybridRetriever）→ 构造引用约束 Prompt → 调 Qwen 生成；
无 Key 或调用失败时降级为抽取式回答（只用条文原文，不杜撰）。

本服务不依赖缓存与数据库——Redis 缓存和查询日志在 API 层
（app/api/qa.py，M4/M5）接入，保证核心链路可独立测试。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

from app.core import llm
from app.core.hybrid_retriever import HybridRetriever
from app.core.prompt import NOT_FOUND_ANSWER, build_prompt

logger = logging.getLogger(__name__)

# 降级模式模拟流式时的每块字符数（视觉上的"打字机"粒度）
_FALLBACK_CHUNK = 12


@dataclass
class QAResult:
    """问答结果（可直接序列化为 API 响应）。"""

    answer: str
    references: list[dict[str, Any]] = field(default_factory=list)
    llm_mode: str = "fallback"
    llm_degraded: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "references": self.references,
            "llm_mode": self.llm_mode,
            "llm_degraded": self.llm_degraded,
        }


def _to_references(docs: list[Document]) -> list[dict[str, Any]]:
    return [
        {
            "code_name": d.metadata["code_name"],
            "code_no": d.metadata["code_no"],
            "code_id": d.metadata["code_id"],
            "is_mandatory": bool(d.metadata.get("is_mandatory")),
            "content": d.metadata.get("raw_content", ""),
        }
        for d in docs
    ]


class QAService:
    """规范问答服务。"""

    def __init__(self, retriever: HybridRetriever | None = None) -> None:
        self.retriever = retriever or HybridRetriever()

    def ask(
        self,
        question: str,
        profession: str | None = None,
        k: int | None = None,
        only_mandatory: bool = False,
    ) -> QAResult:
        question = (question or "").strip()
        if not question:
            return QAResult(
                answer="问题不能为空。",
                llm_mode=llm.llm_mode(),
                llm_degraded=llm.llm_mode() == "fallback",
            )

        # 1. 混合检索
        docs = self.retriever.search(
            question, profession=profession, k=k, only_mandatory=only_mandatory
        )
        references = _to_references(docs)

        # 2. 无命中：按 Prompt 强制规则明确回复
        if not docs:
            return QAResult(
                answer=NOT_FOUND_ANSWER,
                references=[],
                llm_mode=llm.llm_mode(),
                llm_degraded=llm.llm_mode() == "fallback",
            )

        mode = llm.llm_mode()

        # 3. 无 Key：抽取式降级
        if mode == "fallback":
            answer = llm.extractive_answer(docs)
            return QAResult(
                answer=answer,
                references=references,
                llm_mode=mode,
                llm_degraded=True,
            )

        # 4. 真实 LLM；调用失败回落抽取式，并标记 degraded
        prompt = build_prompt(docs, question)
        try:
            answer = llm.chat(prompt)
            return QAResult(
                answer=answer,
                references=references,
                llm_mode="qwen",
                llm_degraded=False,
            )
        except Exception as e:
            logger.warning("Qwen 调用失败，回落抽取式回答: %s", e)
            return QAResult(
                answer=llm.extractive_answer(docs),
                references=references,
                llm_mode="fallback",
                llm_degraded=True,
            )

    # ------------------------------------------------------------------
    # 流式问答
    # ------------------------------------------------------------------
    def ask_stream(
        self,
        question: str,
        profession: str | None = None,
        k: int | None = None,
        only_mandatory: bool = False,
    ):
        """流式问答生成器，事件协议：

        1. 首个事件 {"type": "meta", "references": [...],
            "llm_mode": ..., "llm_degraded": ..., "preset_answer": str|None}
           - preset_answer 非空表示无需流式（空问题/未命中），直接整句展示；
        2. 随后零到多个 {"type": "token", "delta": "..."}；
        3. 结束事件 {"type": "done"}。

        Qwen 模式逐 token 返回；降级模式把抽取式答案切块模拟流式；
        Qwen 流式中途失败则回落抽取式答案继续输出。
        """
        question = (question or "").strip()

        def _emit_meta(
            refs, mode: str, degraded: bool, preset: str | None = None
        ) -> dict[str, Any]:
            return {
                "type": "meta",
                "references": refs,
                "llm_mode": mode,
                "llm_degraded": degraded,
                "preset_answer": preset,
            }

        # 空问题
        if not question:
            yield _emit_meta(
                [], llm.llm_mode(), llm.llm_mode() == "fallback", "问题不能为空。"
            )
            yield {"type": "done"}
            return

        # 检索（流式前一次性完成，引用随 meta 下发）
        docs = self.retriever.search(
            question, profession=profession, k=k, only_mandatory=only_mandatory
        )
        references = _to_references(docs)

        if not docs:
            yield _emit_meta(
                [],
                llm.llm_mode(),
                llm.llm_mode() == "fallback",
                NOT_FOUND_ANSWER,
            )
            yield {"type": "done"}
            return

        mode = llm.llm_mode()
        yield _emit_meta(references, mode, mode == "fallback")

        if mode == "fallback":
            text = llm.extractive_answer(docs)
            for i in range(0, len(text), _FALLBACK_CHUNK):
                yield {"type": "token", "delta": text[i : i + _FALLBACK_CHUNK]}
            yield {"type": "done"}
            return

        # Qwen 真流式
        prompt = build_prompt(docs, question)
        got_any = False
        try:
            for delta in llm.chat_stream(prompt):
                got_any = True
                yield {"type": "token", "delta": delta}
        except Exception as e:
            logger.warning("Qwen 流式调用失败，回落抽取式回答: %s", e)
            text = llm.extractive_answer(docs)
            if not got_any:
                # 一个字都没收到：整段降级输出
                for i in range(0, len(text), _FALLBACK_CHUNK):
                    yield {"type": "token", "delta": text[i : i + _FALLBACK_CHUNK]}
        yield {"type": "done"}
