"""M2 验证：Embedding 降级 + FAISS 入库 + BM25 混合检索。"""
from app.core.hybrid_retriever import HybridRetriever
from app.core.splitter import BuildingCodeSplitter
from app.core.vector_store import VectorStoreManager

# ------------------------------------------------------------------
# 构造规范条文（直接用 splitter 产出，贴近真实入库链路）
# ------------------------------------------------------------------
FIRE_TEXT = """1 总则
1.0.1 为预防建筑火灾制定本规范。
2 耐火等级
2.1.1 厂房和仓库的耐火等级分为一二三四级，并应符合构件燃烧性能要求。
2.1.2 【强制性条文】重要公共建筑的耐火等级**不应低于二级**。
3 防火间距
3.1.1 民用建筑之间的防火间距应根据耐火等级和建筑高度确定。
3.1.2 甲类厂房与重要公共建筑的防火间距严禁小于五十米。
"""

ELEC_TEXT = """1 配电
1.1.1 低压配电线路的短路保护电器应能切断短路电流。
1.1.2 【强制性条文】配电箱**严禁**安装在可燃材料表面。
"""


def _ingest_fire():
    docs = BuildingCodeSplitter("消防", "建筑设计防火规范", "GB50016-DEMO").split(
        FIRE_TEXT, source_file="fire.txt"
    )
    return VectorStoreManager().add_documents(docs)


def _ingest_elec():
    docs = BuildingCodeSplitter("电气", "配电设计演示规范", "GBDEMO-E").split(
        ELEC_TEXT, source_file="elec.txt"
    )
    return VectorStoreManager().add_documents(docs)


# ------------------------------------------------------------------
# 入库
# ------------------------------------------------------------------
def test_ingest_count(temp_index):
    n = _ingest_fire()
    assert n == 5
    assert VectorStoreManager().size == 5


def test_empty_index_search_returns_empty(temp_index):
    r = HybridRetriever()
    assert r.search("任何问题") == []


# ------------------------------------------------------------------
# BM25 / 语义召回（降级 embedder 下两路都依赖词面重合）
# ------------------------------------------------------------------
def test_keyword_recall_fire_resistance(temp_index):
    _ingest_fire()
    r = HybridRetriever()
    hits = r.search("厂房的耐火等级怎么划分")
    assert hits, "应当召回相关条文"
    assert hits[0].metadata["code_id"] == "2.1.1"


def test_bm25_recall_fire_spacing(temp_index):
    _ingest_fire()
    r = HybridRetriever()
    hits = r.search("甲类厂房防火间距限值")
    ids = [d.metadata["code_id"] for d in hits]
    assert "3.1.2" in ids
    # 3.1.2 与查询词面重合最多，应排第一
    assert ids[0] == "3.1.2"


# ------------------------------------------------------------------
# 条文号精确匹配优先
# ------------------------------------------------------------------
def test_code_id_exact_match_first(temp_index):
    _ingest_fire()
    r = HybridRetriever()
    hits = r.search("第3.1.1条对民用建筑有什么规定")
    assert hits, "条文号精确匹配必须命中"
    assert hits[0].metadata["code_id"] == "3.1.1"


# ------------------------------------------------------------------
# 元数据过滤
# ------------------------------------------------------------------
def test_profession_filter(temp_index):
    _ingest_fire()
    _ingest_elec()
    r = HybridRetriever()
    hits = r.search("配电箱安装要求", profession="电气")
    assert hits
    assert all(d.metadata["profession"] == "电气" for d in hits)
    assert hits[0].metadata["code_id"] == "1.1.2"

    # 消防专业下查配电相关词，不应返回电气条文
    fire_hits = r.search("配电箱", profession="消防")
    assert all(d.metadata["profession"] == "消防" for d in fire_hits)


def test_only_mandatory_filter(temp_index):
    _ingest_fire()
    _ingest_elec()
    r = HybridRetriever()
    hits = r.search("建筑防火与配电", only_mandatory=True, k=10)
    assert hits
    assert all(d.metadata["is_mandatory"] for d in hits)
    ids = {d.metadata["code_id"] for d in hits}
    assert {"2.1.2", "1.1.2"}.issubset(ids)


def test_top_k_limit(temp_index):
    _ingest_fire()
    r = HybridRetriever()
    assert len(r.search("建筑", k=2)) <= 2


# ------------------------------------------------------------------
# 增量更新：按规范号删除旧条文
# ------------------------------------------------------------------
def test_delete_by_code_no_and_reingest(temp_index):
    _ingest_fire()
    _ingest_elec()
    vs = VectorStoreManager()
    assert vs.size == 7
    removed = vs.delete_by_code_no("GB50016-DEMO")
    assert removed == 5
    assert vs.size == 2
    # 重新入库同一规范，条数恢复，不产生重复
    _ingest_fire()
    assert vs.size == 7


# ------------------------------------------------------------------
# 持久化：新实例从磁盘加载索引
# ------------------------------------------------------------------
def test_persistence_reload(temp_index):
    _ingest_fire()
    # 丢弃单例，模拟进程重启
    VectorStoreManager._instance = None
    vs2 = VectorStoreManager()
    assert vs2.size == 5
    r = HybridRetriever()
    hits = r.search("甲类厂房防火间距")
    assert hits
    assert hits[0].metadata["code_id"] == "3.1.2"
