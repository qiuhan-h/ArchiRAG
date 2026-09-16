"""pytest 公共夹具：每个测试使用独立的项目内临时 FAISS 索引目录。

注：不使用 pytest 自带 tmp_path——它默认指向系统 Temp，在受限
沙箱环境下不可写。统一放到项目内 tests/.tmp_indexes/ 下。
"""
import shutil
from pathlib import Path

import pytest

from app.config import settings
from app.core import vector_store as vs_module
from app.core.splitter import BuildingCodeSplitter
from app.core.vector_store import VectorStoreManager

TEST_INDEX_ROOT = Path(__file__).resolve().parent / ".tmp_indexes"


@pytest.fixture(autouse=True)
def _force_dev_env(monkeypatch):
    """强制测试环境为 dev + 无 LLM Key，避免 .env 中的生产配置干扰测试。"""
    monkeypatch.setattr(settings, "APP_ENV", "dev")
    monkeypatch.setattr(settings, "QWEN_API_KEY", "")
    # 重置 LLM 单例
    from app.core import llm as llm_module
    llm_module._qwen = None
    llm_module._qwen_tried = False
    # 重置 cache 单例
    from app.core import cache as cache_module
    cache_module._cache = None
    cache_module._cache_type = ""
    yield


@pytest.fixture
def temp_index(request, monkeypatch):
    """把 FAISS 索引目录指向项目内临时路径，并重置单例。"""
    index_dir = TEST_INDEX_ROOT / request.node.name
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "FAISS_INDEX_PATH", str(index_dir))
    vs_module.VectorStoreManager._instance = None
    yield index_dir
    vs_module.VectorStoreManager._instance = None
    shutil.rmtree(index_dir, ignore_errors=True)


_CODE_TEXT = """1 防火
1.1.1 普通条文，描述一般防火要求。
1.1.2 【强制性条文】高层建筑**必须**设置自动喷水灭火系统。
"""


@pytest.fixture
def index_with_code(temp_index):
    """入库 2 条测试条文（1 普通 + 1 强条），供检索/问答测试共用。"""
    docs = BuildingCodeSplitter("消防", "演示防火规范", "GB-M3").split(
        _CODE_TEXT, source_file="m3.txt"
    )
    VectorStoreManager().add_documents(docs)
    return temp_index
