"""FAISS 向量库封装（严格按项目方案：本地索引 + 持久化 + 元数据过滤）。

相对方案示例的工程补强：
- 单例管理，首次访问惰性加载；
- Embedding 均为 L2 归一化向量，使用 IndexFlatIP（内积 == 余弦相似度）；
- delete_by_code_no：同一规范重新入库前先删旧条文，支撑增量更新；
- all_documents：供 BM25 倒排库构建；
- 空索引时检索返回空列表而不是报错。
"""
from __future__ import annotations

import logging
import os
import pickle
import threading
from typing import List

import faiss
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import settings
from app.core.embedder import (
    fallback_lexical_match,
    get_embedder,
    using_fallback,
)

logger = logging.getLogger(__name__)

# 语义命中阈值：向量均已 L2 归一化，IndexFlatIP 的分数即余弦相似度。
# - 真实 bge 模式：只剔除零/负相关（无关文本余弦通常仍为正 0.1~0.3，
#   高阈值会误杀，噪声交给 RRF 与 BM25 通道抑制）；
# - 降级字符哈希模式：哈希桶碰撞不可避免，余弦分不可信，改用确定性
#   词面重合闸 fallback_lexical_match（要求共享 bigram/条文号等强 token）。
SEMANTIC_MIN_SCORE_BGE = 1e-6


class VectorStoreManager:
    """FAISS 索引单例管理器。"""

    _instance: "VectorStoreManager | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "VectorStoreManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._initialized = False
                    cls._instance = inst
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self.embeddings = get_embedder()
        self.index_path = str(settings.faiss_index_abs)
        self.store: FAISS | None = None
        self._load()
        self._initialized = True

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def _load(self) -> None:
        index_file = os.path.join(self.index_path, "index.faiss")
        if os.path.exists(index_file):
            logger.info("加载已存在的 FAISS 索引: %s", self.index_path)
            # 不走 langchain 的 load_local：其内部 faiss.read_index 用
            # C++ fopen，Windows 中文路径（如"解压盘"）会失败。
            # 改为 Python I/O 读字节 + deserialize_index。
            with open(index_file, "rb") as f:
                index = faiss.deserialize_index(
                    np.frombuffer(f.read(), dtype=np.uint8)
                )
            with open(os.path.join(self.index_path, "index.pkl"), "rb") as f:
                docstore, index_to_docstore_id = pickle.load(f)
            self.store = FAISS(
                self.embeddings, index, docstore, index_to_docstore_id
            )
        else:
            logger.info("FAISS 索引不存在，将在首次入库时创建: %s", self.index_path)

    def _save(self) -> None:
        os.makedirs(self.index_path, exist_ok=True)
        assert self.store is not None
        # 同样绕过 faiss.write_index 的 C++ fopen（不支持 Windows 中文路径），
        # 序列化为字节后用 Python 文件 I/O 落盘。
        serialized = faiss.serialize_index(self.store.index).tobytes()
        with open(os.path.join(self.index_path, "index.faiss"), "wb") as f:
            f.write(serialized)
        with open(os.path.join(self.index_path, "index.pkl"), "wb") as f:
            pickle.dump(
                (self.store.docstore, self.store.index_to_docstore_id), f
            )

    # ------------------------------------------------------------------
    # 写入 / 删除
    # ------------------------------------------------------------------
    def add_documents(self, docs: List[Document]) -> int:
        """增量写入条文并持久化，返回写入条数。"""
        if not docs:
            return 0
        if self.store is None:
            # 显式建内积索引：Embedding 均为 L2 归一化向量，
            # 内积 == 余弦相似度（langchain 默认的 L2 索引无法按余弦阈值过滤）。
            dim = len(self.embeddings.embed_query("dimension_probe"))
            index = faiss.IndexFlatIP(dim)
            self.store = FAISS(
                self.embeddings, index, InMemoryDocstore(), {}
            )
        self.store.add_documents(docs)
        self._save()
        logger.info("FAISS 入库 %d 条，索引目录: %s", len(docs), self.index_path)
        return len(docs)

    def delete_by_code_no(self, code_no: str) -> int:
        """删除某本规范的全部条文（规范更新重新入库前调用）。"""
        if self.store is None:
            return 0
        ids_to_delete = [
            doc_id
            for doc_id, doc in self.store.docstore._dict.items()
            if doc.metadata.get("code_no") == code_no
        ]
        if not ids_to_delete:
            return 0
        self.store.delete(ids_to_delete)
        self._save()
        logger.info("规范 %s 删除旧条文 %d 条", code_no, len(ids_to_delete))
        return len(ids_to_delete)

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        if self.store is None:
            return 0
        return len(self.store.docstore._dict)

    def all_documents(self) -> List[Document]:
        if self.store is None:
            return []
        return list(self.store.docstore._dict.values())

    def similarity_search_with_filter(
        self,
        query: str,
        k: int,
        profession: str | None = None,
        only_mandatory: bool = False,
    ) -> List[Document]:
        """语义检索（可选专业过滤）。

        only_mandatory 在混合检索层统一处理（需要跨两路结果合并后过滤，
        避免每路各自过滤破坏融合排序）。
        """
        if self.store is None or self.size == 0:
            return []
        # fetch 多一些，过滤后仍有足够候选；上限不超过索引总量
        fetch_k = min(k * 2 if profession else k, self.size)
        filter_dict = {"profession": profession} if profession else None
        pairs = self.store.similarity_search_with_score(
            query, k=fetch_k, filter=filter_dict
        )
        if using_fallback():
            # 哈希向量碰撞不可避免，且余弦只反映词面重合：
            # 用确定性词面命中闸（共享 bigram/条文号等强 token）做校验
            return [
                doc
                for doc, score in pairs
                if float(score) > 0
                and fallback_lexical_match(query, doc.page_content)
            ]
        # 真实 bge：按余弦阈值过滤
        return [
            doc
            for doc, score in pairs
            if float(score) > SEMANTIC_MIN_SCORE_BGE
        ]
