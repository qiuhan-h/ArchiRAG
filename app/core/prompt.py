"""Prompt 约束 —— 杜绝杜撰的关键模块。

严格按项目开发方案：
1. 必须逐字引用条文原文，格式：《规范名称》第X.X.X条：原文内容；
2. 严禁编造、改写、推测任何条文内容；
3. 条文中没有答案时，必须明确回复"现有规范库中未找到相关条文"；
4. 强制性条文显著标注【强制性条文】；
5. 回答末尾列出所有引用条文编号。
"""
from __future__ import annotations

from langchain_core.documents import Document

SYSTEM_PROMPT = """你是一名资深建筑规范专家。请严格依据提供的条文回答用户问题。

【强制规则】
1. 必须逐字引用条文原文，格式为：《规范名称》第X.X.X条：原文内容
2. 严禁编造、改写、推测任何条文内容
3. 若提供的条文中没有答案，必须明确回复："现有规范库中未找到相关条文"
4. 若涉及强制性条文，需在回答中显著标注【强制性条文】
5. 回答末尾列出所有引用条文的编号

【参考条文】
{context}

【用户问题】
{question}
"""

NOT_FOUND_ANSWER = "现有规范库中未找到相关条文。"


def format_clause(doc: Document) -> str:
    """把单条检索结果格式化为喂给 LLM 的参考条文。"""
    md = doc.metadata
    strong = "【强制性条文】" if md.get("is_mandatory") else ""
    head = f"《{md['code_name']}》第{md['code_id']}条{strong}"
    return f"{head}：\n{md.get('raw_content', '')}"


def build_prompt(context_docs: list[Document], question: str) -> str:
    """构造完整 Prompt（context 为混合检索命中的条文）。"""
    ctx = "\n\n".join(format_clause(d) for d in context_docs)
    return SYSTEM_PROMPT.format(context=ctx, question=question)
