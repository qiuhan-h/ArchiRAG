"""Query 改写 —— 用 LLM 把口语问题转为规范术语，提升召回率。

用户口语："高层住宅几个楼梯" → 规范术语："高层住宅建筑安全出口数量要求 疏散楼梯"

策略：
- Qwen 模式：调 LLM 改写，输出扩展后的检索 query（原始关键词 + 规范术语）；
- 降级模式：无 Key 时跳过，用原始 query 检索（不阻塞链路）；
- 改写失败：静默回落原始 query，不影响主链路。

改写后的 query 与原始 query 分别走一次三路混合检索，
两轮结果合并后 RRF 融合，最大化召回。
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# 改写 Prompt：要求输出规范术语 + 同义词扩展，一句话
_REWRITE_SYSTEM = """你是建筑规范检索专家。请将用户的口语化问题改写为规范术语关键词，用于检索建筑规范条文库。

要求：
1. 保留原始问题的核心意图
2. 转换为规范中常用的正式术语（如"几个楼梯"→"安全出口数量 疏散楼梯"）
3. 补充同义词或相关术语（如"防火间距"也加上"防火距离"）
4. 只输出改写后的关键词，不要解释，不要标点，用空格分隔
5. 如果问题已经是规范术语，原样返回即可

示例：
用户：高层住宅几个楼梯
改写：高层住宅 安全出口 疏散楼梯 数量要求

用户：防火墙能不能开门
改写：防火墙 洞口 门窗 防火分隔

用户：第3.1.2条
改写：3.1.2 条文内容

用户：排烟系统维护
改写：排烟系统 维护保养 定期检查
"""

# 条文号模式：如果 query 基本就是条文号，不改写
_CODE_ID_ONLY = re.compile(r"^[\d.]+\s*$|^[A-Z]\.\d+\.\d+\s*$")


def rewrite_query(question: str) -> str:
    """用 LLM 改写 query，返回扩展后的检索词串。

    - 无 Qwen Key 或调用失败：返回原始 question（不阻塞）
    - 纯条文号查询：直接返回（无需改写）
    - 成功：返回 "原始问题 改写后的规范术语" 拼接串
    """
    question = question.strip()
    if not question:
        return question

    # 纯条文号不改写
    if _CODE_ID_ONLY.match(question):
        return question

    # 降级模式：无 Qwen Key，跳过
    from app.core import llm

    if llm.llm_mode() != "qwen":
        return question

    try:
        from app.config import settings

        qwen = llm._get_qwen()
        if qwen is None:
            return question

        resp = qwen._client.chat.completions.create(
            model=settings.QWEN_MODEL,
            messages=[
                {"role": "system", "content": _REWRITE_SYSTEM},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=100,
        )
        rewritten = (resp.choices[0].message.content or "").strip()

        if not rewritten or rewritten == question:
            return question

        # 拼接原始 + 改写，最大化召回
        combined = f"{question} {rewritten}"
        logger.info("Query 改写: %r -> %r", question, combined)
        return combined

    except Exception as e:
        logger.warning("Query 改写失败（%s），使用原始 query", e)
        return question
