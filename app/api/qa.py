"""问答接口 /api/qa —— 含缓存 + DB 日志 + SSE 流式。"""
from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.core.cache import cache_get, cache_set, make_cache_key
from app.core.qa_service import QAService
from app.db import crud
from app.schemas.models import (
    QARequest,
    QAResponse,
    ReferenceItem,
    StreamQARequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/qa", tags=["qa"])

_qa_service: QAService | None = None


def _get_service() -> QAService:
    global _qa_service
    if _qa_service is None:
        _qa_service = QAService()
    return _qa_service


@router.post("", response_model=QAResponse)
def ask(req: QARequest) -> QAResponse:
    """规范问答：检索 → LLM → 结构化引用（带缓存）。"""
    # 1. 查缓存
    cache_key = make_cache_key(req.question, req.profession, req.k)
    cached = cache_get(cache_key)
    if cached is not None:
        data = json.loads(cached)
        data["cache_hit"] = True
        # 记录缓存命中
        crud.log_query(
            req.question,
            data["answer"],
            data.get("references", []),
            data.get("llm_mode", "fallback"),
            data.get("llm_degraded", True),
            cache_hit=True,
            elapsed_ms=0,
        )
        return QAResponse(**data)

    # 2. 正常问答
    t0 = time.time()
    result = _get_service().ask(
        question=req.question,
        profession=req.profession,
        k=req.k,
        only_mandatory=req.only_mandatory,
    )
    elapsed_ms = int((time.time() - t0) * 1000)

    # 3. 写缓存
    resp_data = {
        "answer": result.answer,
        "references": result.references,
        "llm_mode": result.llm_mode,
        "llm_degraded": result.llm_degraded,
    }
    cache_set(cache_key, json.dumps(resp_data, ensure_ascii=False))

    # 4. 写 DB 日志
    crud.log_query(
        req.question,
        result.answer,
        result.references,
        result.llm_mode,
        result.llm_degraded,
        cache_hit=False,
        elapsed_ms=elapsed_ms,
    )

    return QAResponse(
        answer=result.answer,
        references=[ReferenceItem(**r) for r in result.references],
        llm_mode=result.llm_mode,
        llm_degraded=result.llm_degraded,
        cache_hit=False,
        elapsed_ms=elapsed_ms,
    )


def _sse(obj: dict) -> str:
    """格式化一条 SSE 事件。"""
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@router.post("/stream")
def ask_stream(req: StreamQARequest) -> StreamingResponse:
    """流式问答（SSE）：meta → token* → done，同时持久化会话。"""
    question = req.question.strip()
    cache_key = make_cache_key(question, req.profession, req.k)

    def event_gen():
        t0 = time.time()

        # 1. 缓存命中：整句通过 preset_answer 一次下发
        cached = cache_get(cache_key)
        if cached is not None:
            data = json.loads(cached)
            sid = req.session_id or crud.create_session(question[:20])
            answer = data["answer"]
            references = data.get("references", [])
            yield _sse({
                "type": "meta",
                "session_id": sid,
                "references": references,
                "llm_mode": data.get("llm_mode", "fallback"),
                "llm_degraded": data.get("llm_degraded", True),
                "cache_hit": True,
                "preset_answer": answer,
            })
            elapsed_ms = int((time.time() - t0) * 1000)
            try:
                crud.add_message(sid, "user", question)
                crud.add_message(
                    sid, "assistant", answer, references=references,
                    llm_mode=data.get("llm_mode"), cache_hit=True,
                    elapsed_ms=elapsed_ms,
                )
                crud.log_query(
                    question, answer, references,
                    data.get("llm_mode", "fallback"),
                    data.get("llm_degraded", True),
                    cache_hit=True, elapsed_ms=elapsed_ms,
                )
            except Exception:
                logger.exception("缓存命中路径持久化失败")
            yield _sse({"type": "done", "elapsed_ms": elapsed_ms})
            return

        # 2. 正常流式
        sid = req.session_id or crud.create_session(question[:20])
        try:
            crud.add_message(sid, "user", question)
            if req.session_id is not None:
                crud.touch_session(sid)
        except Exception:
            logger.exception("写入用户消息失败")

        answer_parts: list[str] = []
        meta_info: dict = {}
        preset: str | None = None

        for ev in _get_service().ask_stream(
            question=question,
            profession=req.profession,
            k=req.k,
            only_mandatory=req.only_mandatory,
        ):
            etype = ev.get("type")
            if etype == "meta":
                meta_info = ev
                preset = ev.get("preset_answer")
                yield _sse({**ev, "session_id": sid, "cache_hit": False})
            elif etype == "token":
                answer_parts.append(ev["delta"])
                yield _sse(ev)
            # done 事件在持久化完成后再下发

        answer = preset if preset is not None else "".join(answer_parts)
        elapsed_ms = int((time.time() - t0) * 1000)
        references = meta_info.get("references", [])
        llm_mode = meta_info.get("llm_mode", "fallback")
        llm_degraded = meta_info.get("llm_degraded", True)

        # 3. 持久化 assistant 消息 + 查询日志 + 缓存
        try:
            crud.add_message(
                sid, "assistant", answer, references=references,
                llm_mode=llm_mode, cache_hit=False, elapsed_ms=elapsed_ms,
            )
            crud.log_query(
                question, answer, references, llm_mode, llm_degraded,
                cache_hit=False, elapsed_ms=elapsed_ms,
            )
            cache_set(
                cache_key,
                json.dumps(
                    {
                        "answer": answer,
                        "references": references,
                        "llm_mode": llm_mode,
                        "llm_degraded": llm_degraded,
                    },
                    ensure_ascii=False,
                ),
            )
        except Exception:
            logger.exception("流式路径持久化失败")

        yield _sse({"type": "done", "elapsed_ms": elapsed_ms})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
        },
    )
