"""混合检索 —— 项目命中率的关键模块。

三路召回 + RRF（Reciprocal Rank Fusion）融合：

1. 语义通道：FAISS 向量相似度（bge 真实向量 / 降级哈希向量）；
2. BM25 通道：rank_bm25.BM25Okapi 对全部条文建倒排索引（中文按
   字 unigram/bigram + 条文号 + 英文词切分），按分数取 Top-N；
3. 条文号精确通道：query 中出现 "3.2.1" 等编号时，按 code_id 精确
   命中（规范问答的高频形态，必须排最前）。

融合：每路给出排名，score = Σ 1/(RRF_K + rank)，多路同时命中的
条文自然排前；最后按专业元数据过滤、按 only_mandatory 过滤。

说明：方案示例 import 了 BM25Okapi 但未使用、关键词通道靠遍历
docstore，这里落地为真正的倒排检索，10 万条文规模下查询为 O(候选数)。
"""
from __future__ import annotations

import logging
import re
from typing import List

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.config import settings
from app.core.embedder import fallback_lexical_match, tokenize_for_fallback, using_fallback
from app.core.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class HybridRetriever:
    """语义向量 + BM25 关键词 + 条文号精确匹配的混合检索器。"""

    CODE_ID_PATTERN = re.compile(r"\d+(?:\.\d+){1,3}|[A-Z]\.\d+\.\d+")
    RRF_K = 60  # RRF 常数，越大则排名靠后的结果权重衰减越平缓

    def __init__(self) -> None:
        self.vs = VectorStoreManager()
        self._bm25: BM25Okapi | None = None
        self._bm25_docs: List[Document] = []
        self._indexed_size = -1

    # ------------------------------------------------------------------
    # BM25 索引（惰性构建；入库后条数变化时自动重建）
    # ------------------------------------------------------------------
    def _ensure_bm25(self, profession: str | None) -> None:
        docs = self.vs.all_documents()
        if profession:
            docs = [d for d in docs if d.metadata.get("profession") == profession]
        # 未过滤的总条数用于感知"是否有新入库"
        total = self.vs.size
        if self._bm25 is not None and total == self._indexed_size and not profession:
            return
        corpus_tokens = [tokenize_for_fallback(d.page_content) for d in docs]
        if corpus_tokens and any(corpus_tokens):
            self._bm25 = BM25Okapi(corpus_tokens)
        else:
            self._bm25 = None
        self._bm25_docs = docs
        self._indexed_size = total

    def _bm25_search(
        self, query: str, k: int, profession: str | None
    ) -> List[Document]:
        self._ensure_bm25(profession)
        if self._bm25 is None or not self._bm25_docs:
            return []
        scores = self._bm25.get_scores(tokenize_for_fallback(query))
        # 取分数 > 0 的 Top-k
        ranked = sorted(
            zip(self._bm25_docs, scores), key=lambda x: float(x[1]), reverse=True
        )
        hits = [doc for doc, score in ranked if float(score) > 0]
        if using_fallback():
            # 降级模式下 BM25 会给单字重合（承/要/构…）正分，
            # 统一用词面强 token 闸过滤噪声（bge 模式保留完整 BM25 召回）
            hits = [d for d in hits if fallback_lexical_match(query, d.page_content)]
        return hits[:k]

    # ------------------------------------------------------------------
    # 条文号精确通道
    # ------------------------------------------------------------------
    def _code_id_search(
        self, code_ids: List[str], profession: str | None
    ) -> List[Document]:
        wanted = set(code_ids)
        hits: List[Document] = []
        for d in self.vs.all_documents():
            if profession and d.metadata.get("profession") != profession:
                continue
            if d.metadata.get("code_id") in wanted:
                hits.append(d)
        return hits

    # ------------------------------------------------------------------
    # RRF 融合
    # ------------------------------------------------------------------
    @staticmethod
    def _doc_key(d: Document) -> tuple[str, str]:
        return (d.metadata.get("code_no", ""), d.metadata.get("code_id", ""))

    def _rrf_fuse(
        self, ranked_lists: List[List[Document]], k: int
    ) -> List[Document]:
        scores: dict[tuple[str, str], float] = {}
        store: dict[tuple[str, str], Document] = {}
        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked):
                key = self._doc_key(doc)
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.RRF_K + rank + 1)
                store.setdefault(key, doc)
        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [store[key] for key, _ in ordered[:k]]

    # ------------------------------------------------------------------
    # 公开入口
    # ------------------------------------------------------------------
    def _single_search(
        self, query: str, k: int, profession: str | None
    ) -> List[List[Document]]:
        """单轮三路召回，返回各路 ranked list（不做融合）。"""
        semantic_docs = self.vs.similarity_search_with_filter(
            query, k=k * 2, profession=profession
        )
        bm25_docs = self._bm25_search(query, k=k * 2, profession=profession)

        ranked_lists = [semantic_docs, bm25_docs]

        code_ids = self.CODE_ID_PATTERN.findall(query)
        if code_ids:
            exact_docs = self._code_id_search(code_ids, profession)
            ranked_lists.insert(0, exact_docs)

        return ranked_lists

    def search(
        self,
        query: str,
        profession: str | None = None,
        k: int | None = None,
        only_mandatory: bool = False,
    ) -> List[Document]:
        k = k or settings.TOP_K
        query = query.strip()
        if not query:
            return []

        # Query 改写：LLM 把口语转为规范术语，扩展检索词
        from app.core.query_rewriter import rewrite_query

        rewritten = rewrite_query(query)
        is_rewritten = rewritten != query

        # 原始 query 三路召回
        ranked_lists = self._single_search(query, k, profession)

        # 改写后 query 再做一轮三路召回，合并到 RRF 融合
        if is_rewritten:
            rewritten_lists = self._single_search(rewritten, k, profession)
            ranked_lists.extend(rewritten_lists)

        # only_mandatory 时放宽召回数量，过滤后再截断
        fetch_limit = k * 4 if only_mandatory else k
        merged = self._rrf_fuse(ranked_lists, k=fetch_limit)

        if only_mandatory:
            merged = [d for d in merged if d.metadata.get("is_mandatory")]
        return merged[:k]
