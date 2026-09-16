"""M3 验证：Prompt 引用约束 + LLM 降级 + 问答主链路。"""
import pytest

from app.core import llm
from app.core.prompt import (
    NOT_FOUND_ANSWER,
    build_prompt,
    format_clause,
)
from app.core.qa_service import QAService
from app.core.splitter import BuildingCodeSplitter


# ------------------------------------------------------------------
# Prompt 构造
# ------------------------------------------------------------------
def test_prompt_contains_forced_rules(index_with_code):
    from app.core.hybrid_retriever import HybridRetriever

    docs = HybridRetriever().search("自动喷水灭火", k=2)
    prompt = build_prompt(docs, "高层建筑是否必须设喷淋？")
    # 五条强制规则关键约束
    assert "必须逐字引用条文原文" in prompt
    assert "严禁编造" in prompt
    assert "现有规范库中未找到相关条文" in prompt
    assert "强制性条文" in prompt
    assert "高层建筑是否必须设喷淋？" in prompt


def test_format_clause_strong_tag():
    docs = BuildingCodeSplitter("消防", "规范N", "GB-X").split(
        "1.1.1 【强制性条文】**必须**做某事。", "x.txt"
    )
    text = format_clause(docs[0])
    assert "《规范N》第1.1.1条【强制性条文】" in text
    assert "必须" in text


# ------------------------------------------------------------------
# 降级 LLM：抽取式回答不杜撰
# ------------------------------------------------------------------
def test_extractive_answer_quotes_verbatim(index_with_code):
    from app.core.hybrid_retriever import HybridRetriever

    docs = HybridRetriever().search("自动喷水灭火系统", k=1)
    answer = llm.extractive_answer(docs)
    # 逐字引用原文
    assert "《演示防火规范》第1.1.2条" in answer
    assert "自动喷水灭火系统" in answer
    assert "【强制性条文】" in answer
    # 末尾引用编号
    assert "第1.1.2条" in answer
    # 降级标注
    assert "降级模式" in answer


def test_extractive_answer_empty():
    assert llm.extractive_answer([]) == NOT_FOUND_ANSWER


def test_llm_mode_is_fallback_without_key():
    # CI/开发环境没有 .env / Key
    assert llm.llm_mode() == "fallback"


def test_chat_raises_without_key():
    with pytest.raises(RuntimeError):
        llm.chat("任意 prompt")


# ------------------------------------------------------------------
# 降级词法噪声闸：单字重合不算命中，bigram/条文号才算
# ------------------------------------------------------------------
def test_lexical_gate_requires_strong_token():
    from app.core.embedder import fallback_lexical_match as m

    # 仅共享单字"承"（承担 vs 承重）→ 噪声，不算命中
    assert m("检测费用由谁承担", "主要承重构件必须采用不燃材料") is False
    # 共享 bigram"系统" → 命中
    assert m("喷淋系统", "必须设置自动喷水灭火系统") is True
    # 零重合
    assert m("基坑支护锚杆", "防火间距严禁小于五十米") is False
    # 单字短查询：退回 unigram
    assert m("火", "民用建筑之间的防火间距") is True
    # 条文号整体为强 token
    assert m("第3.1.2条怎么规定", "3.1.2 甲类厂房防火间距") is True


# ------------------------------------------------------------------
# 问答主链路（降级模式端到端）
# ------------------------------------------------------------------
def test_qa_hits_and_citations(index_with_code):
    svc = QAService()
    result = svc.ask("高层建筑必须设置什么灭火系统？")
    assert result.llm_mode == "fallback"
    assert result.llm_degraded is True
    assert "自动喷水灭火系统" in result.answer
    assert result.references
    top = result.references[0]
    assert top["code_id"] == "1.1.2"
    assert top["is_mandatory"] is True
    assert top["code_no"] == "GB-M3"


def test_qa_not_found(index_with_code):
    svc = QAService()
    result = svc.ask("基坑支护锚杆长度如何计算")  # 库中完全无关
    assert result.answer == NOT_FOUND_ANSWER
    assert result.references == []


def test_qa_empty_question(index_with_code):
    result = QAService().ask("   ")
    assert "不能为空" in result.answer


def test_qa_only_mandatory(index_with_code):
    result = QAService().ask("防火", only_mandatory=True, k=10)
    assert result.references
    assert all(r["is_mandatory"] for r in result.references)


def test_qa_to_dict_shape(index_with_code):
    d = QAService().ask("普通防火要求").to_dict()
    assert set(d.keys()) == {
        "answer",
        "references",
        "llm_mode",
        "llm_degraded",
    }


# ------------------------------------------------------------------
# 真实 LLM 调用失败时自动回落抽取式
# ------------------------------------------------------------------
def test_qa_falls_back_when_qwen_errors(index_with_code, monkeypatch):
    monkeypatch.setattr(llm, "llm_mode", lambda: "qwen")

    def _boom(prompt, temperature=None):
        raise RuntimeError("simulated qwen outage")

    monkeypatch.setattr(llm, "chat", _boom)
    result = QAService().ask("自动喷水灭火系统")
    assert result.llm_mode == "fallback"
    assert result.llm_degraded is True
    assert "自动喷水灭火系统" in result.answer  # 回落回答仍可用
