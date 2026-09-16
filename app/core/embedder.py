"""Embedding 层（严格按方案：BAAI/bge-large-zh-v1.5，缺失时降级）。

- 真实模式：sentence-transformers 加载 bge-large-zh-v1.5（需安装
  requirements-embed.txt，模型约 1.3GB，首次运行自动下载）。
- 降级模式：确定性"字符 n-gram 哈希向量"。无 torch 依赖、零网络、
  结果稳定可复现，能跑通全链路且对中文有基础相似度（共享汉字
  bigram 越多，余弦相似度越高）。仅用于开发/测试，不代表真实语义效果。
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import List

from langchain_core.embeddings import Embeddings

from app.config import settings

logger = logging.getLogger(__name__)

# 条文号（3.2.1 / A.0.1）作为独立特征，提升编号相关查询的区分度
_CODE_ID_RE = re.compile(r"\d+(?:\.\d+){1,3}|[A-Z]\.\d+\.\d+")
# 连续中文片段
_ZH_RE = re.compile(r"[\u4e00-\u9fa5]+")
# 英文/数字词
_EN_RE = re.compile(r"[A-Za-z0-9]+")


def _hash_token(token: str, dim: int) -> tuple[int, float]:
    """特征哈希：返回 (桶下标, 符号)，signed hashing 降低碰撞偏差。"""
    h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
    return h % dim, 1.0 if (h >> 63) & 1 else -1.0


def tokenize_for_fallback(text: str) -> list[str]:
    """降级模式分词：中文 unigram + bigram + 英文词 + 条文号整体。"""
    tokens: list[str] = list(_CODE_ID_RE.findall(text))
    for seg in _ZH_RE.findall(text):
        tokens.extend(list(seg))  # unigram
        tokens.extend(seg[i : i + 2] for i in range(len(seg) - 1))  # bigram
    tokens.extend(w.lower() for w in _EN_RE.findall(text))
    return tokens


def _is_strong_token(tok: str) -> bool:
    """强 token：中文 bigram、英文/数字词、条文号——有实际区分度。"""
    if len(tok) == 2 and all("\u4e00" <= c <= "\u9fa5" for c in tok):
        return True
    return bool(_EN_RE.fullmatch(tok) or _CODE_ID_RE.fullmatch(tok))


def fallback_lexical_match(query: str, text: str) -> bool:
    """降级模式的词面命中判定（供语义/BM25 两个通道共用的噪声闸）。

    中文单字 unigram（如"承/要/构"）几乎无区分度，单独出现一律不算命中；
    必须至少共享一个强 token（中文 bigram / 英文词 / 条文号）。
    Query 本身短到没有任何强 token（如单字"火"）时，退回 unigram 判定。
    """
    q_set = set(tokenize_for_fallback(query))
    t_set = set(tokenize_for_fallback(text))
    shared = q_set & t_set
    if not shared:
        return False
    if any(_is_strong_token(t) for t in shared):
        return True
    q_strong = [t for t in q_set if _is_strong_token(t)]
    if not q_strong:
        return bool(shared)  # 短查询：单字重合也算
    return False


class FallbackEmbeddings(Embeddings):
    """确定性字符 n-gram 哈希向量（L2 归一化，配合 FAISS 内积索引）。"""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim or settings.FALLBACK_EMBED_DIM

    def _embed_one(self, text: str) -> list[float]:
        import numpy as np

        vec = np.zeros(self.dim, dtype="float32")
        for tok in tokenize_for_fallback(text):
            idx, sign = _hash_token(tok, self.dim)
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


class BgeEmbeddings(Embeddings):
    """bge-large-zh-v1.5 真实 Embedding（薄封装，保持 LangChain 接口一致）。"""

    def __init__(self) -> None:
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings

        logger.info("加载真实 Embedding 模型: %s", settings.EMBEDDING_MODEL)
        self._impl = HuggingFaceBgeEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": settings.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._impl.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._impl.embed_query(text)


_embeddings: Embeddings | None = None


def get_embedder() -> Embeddings:
    """单例工厂：优先 bge，sentence-transformers 不可用时降级。"""
    global _embeddings
    if _embeddings is not None:
        return _embeddings
    try:
        import sentence_transformers  # noqa: F401  仅探测可用性
    except (ImportError, OSError) as e:
        # OSError：torch DLL 加载失败（c10.dll 等）也视为不可用
        logger.warning(
            "sentence-transformers 不可用（%s: %s），Embedding 降级为字符 n-gram 哈希向量；"
            "真实语义检索请执行: pip install -r requirements-embed.txt",
            type(e).__name__,
            e,
        )
        _embeddings = FallbackEmbeddings()
        return _embeddings
    try:
        _embeddings = BgeEmbeddings()
    except Exception as e:  # 模型下载失败等
        logger.warning("bge 模型加载失败（%s），降级为字符 n-gram 哈希向量", e)
        _embeddings = FallbackEmbeddings()
    return _embeddings


def using_fallback() -> bool:
    """当前是否运行在降级 Embedding 模式。"""
    return isinstance(get_embedder(), FallbackEmbeddings)
