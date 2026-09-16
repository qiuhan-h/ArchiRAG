"""LLM 封装（严格按方案：Qwen，DashScope OpenAI 兼容模式）。

- 真实模式：QWEN_API_KEY 存在时，经 OpenAI 兼容协议调用 qwen-plus；
- 降级模式：无 Key（或调用失败）时使用 ExtractiveLLM —— 不生成新文本，
  只用检索命中的条文原文按引用格式拼合回答。它"答得笨"但绝不杜撰，
  保证开发/CI 环境全链路可跑且回答可审计。

问答服务（qa_service）根据 llm_mode() 选择路径；真实调用异常时
自动回落到抽取式回答，并在结果中标记 llm_degraded=True。
"""
from __future__ import annotations

import logging
from typing import List

from langchain_core.documents import Document

from app.config import settings
from app.core.prompt import NOT_FOUND_ANSWER

logger = logging.getLogger(__name__)


class QwenLLM:
    """Qwen（OpenAI 兼容协议）。"""

    def __init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            api_key=settings.QWEN_API_KEY,
            base_url=settings.QWEN_BASE_URL,
        )

    def chat(self, prompt: str, temperature: float | None = None) -> str:
        resp = self._client.chat.completions.create(
            model=settings.QWEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=(
                settings.LLM_TEMPERATURE if temperature is None else temperature
            ),
        )
        return resp.choices[0].message.content or ""

    def chat_stream(self, prompt: str, temperature: float | None = None):
        """流式调用，逐 token yield 文本增量。"""
        stream = self._client.chat.completions.create(
            model=settings.QWEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=(
                settings.LLM_TEMPERATURE if temperature is None else temperature
            ),
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and getattr(delta, "content", None):
                yield delta.content


_qwen: QwenLLM | None = None
_qwen_tried = False


def llm_mode() -> str:
    """当前 LLM 模式：'qwen' 或 'fallback'。"""
    return "qwen" if settings.QWEN_API_KEY else "fallback"


def _get_qwen() -> QwenLLM | None:
    """惰性构造 Qwen 客户端；无 Key 返回 None。"""
    global _qwen, _qwen_tried
    if not settings.QWEN_API_KEY:
        return None
    if _qwen is None and not _qwen_tried:
        _qwen_tried = True
        try:
            _qwen = QwenLLM()
        except Exception as e:  # openai 包缺失等
            logger.warning("Qwen 客户端初始化失败（%s），降级抽取式回答", e)
            return None
    return _qwen


def chat(prompt: str, temperature: float | None = None) -> str:
    """调用 Qwen（与项目开发方案签名一致）。

    无 Key 时抛 RuntimeError——调用方（qa_service）应改走
    extractive_answer 降级路径，而不是直接调用本函数。
    """
    qwen = _get_qwen()
    if qwen is None:
        raise RuntimeError("QWEN_API_KEY 未配置，无法调用真实 LLM")
    return qwen.chat(prompt, temperature)


def chat_stream(prompt: str, temperature: float | None = None):
    """流式调用 Qwen，逐 token yield。无 Key 时抛 RuntimeError。"""
    qwen = _get_qwen()
    if qwen is None:
        raise RuntimeError("QWEN_API_KEY 未配置，无法调用真实 LLM")
    yield from qwen.chat_stream(prompt, temperature)


def extractive_answer(docs: List[Document]) -> str:
    """降级回答：严格用检索条文原文拼合（逐字引用，不生成、不改写）。"""
    if not docs:
        return NOT_FOUND_ANSWER
    lines: list[str] = ["根据规范库检索到以下相关条文："]
    for d in docs:
        md = d.metadata
        strong = "【强制性条文】" if md.get("is_mandatory") else ""
        lines.append(
            f"《{md['code_name']}》第{md['code_id']}条{strong}："
            f"{md.get('raw_content', '')}"
        )
    refs = "、".join(f"第{d.metadata['code_id']}条" for d in docs)
    lines.append(f"引用条文编号：{refs}")
    lines.append("（当前为无 LLM Key 的降级模式，以上为条文原文摘录，未作改写）")
    return "\n".join(lines)
