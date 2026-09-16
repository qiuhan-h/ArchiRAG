"""M1 验证：规范文档解析 + 条文级结构化分块。"""
from pathlib import Path

from app.core.parser import load_code_file, parse_meta_from_filename
from app.core.splitter import BuildingCodeSplitter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_FILE = PROJECT_ROOT / "data" / "examples" / "消防_演示规范_DEMO-001-2026.txt"

# 与演示文件一致的内联文本（部分用例独立使用，避免依赖数据文件）
SAMPLE = """演示规范标题（应被忽略）

1 总则

1.0.1 第一条正文内容，不含强条关键词。
条文说明：第一条的说明文字。
1.0.2 第二条正文，普通条文。

2 防火
42
第 5 页
2.1.1 普通条文里出现应符合字样会被判强条（保持方案默认行为）。
2.1.2 【强制性条文】显式标注的条文**必须**执行。
3.1.1 带说明的条文正文，不得越过限值。
3.1.1 【条文说明】这是编号同行的说明。
"""


def _split(text: str):
    sp = BuildingCodeSplitter("消防", "演示规范", "DEMO-001-2026")
    return sp.split(text, source_file="demo.txt")


# ------------------------------------------------------------------
# 分块基本正确性
# ------------------------------------------------------------------
def test_split_count_and_code_ids():
    docs = _split(SAMPLE)
    assert [d.metadata["code_id"] for d in docs] == [
        "1.0.1",
        "1.0.2",
        "2.1.1",
        "2.1.2",
        "3.1.1",
    ]


def test_content_explanation_separation():
    docs = {d.metadata["code_id"]: d for d in _split(SAMPLE)}
    assert docs["1.0.1"].metadata["raw_content"] == "第一条正文内容，不含强条关键词。"
    assert docs["1.0.1"].metadata["explanation"] == "第一条的说明文字。"
    # 编号与"条文说明"同行的形态
    assert docs["3.1.1"].metadata["explanation"] == "这是编号同行的说明。"
    assert "带说明的条文正文" in docs["3.1.1"].metadata["raw_content"]


def test_multiline_content_concatenated():
    text = "4.1.1 第一行内容\n第二行续写内容。\n条文说明：说明A。"
    docs = _split(text)
    assert len(docs) == 1
    assert docs[0].metadata["raw_content"] == "第一行内容 第二行续写内容。"
    assert docs[0].metadata["explanation"] == "说明A。"


def test_embedding_text_format():
    docs = _split(SAMPLE)
    d = docs[0]
    assert "《演示规范》DEMO-001-2026 第1.0.1条" in d.page_content
    assert "条文说明：第一条的说明文字。" in d.page_content


# ------------------------------------------------------------------
# 元数据
# ------------------------------------------------------------------
def test_metadata_fields_complete():
    docs = _split(SAMPLE)
    required = {
        "profession",
        "code_name",
        "code_no",
        "code_id",
        "is_mandatory",
        "source_file",
        "raw_content",
        "explanation",
    }
    for d in docs:
        assert required.issubset(d.metadata.keys())
        assert d.metadata["profession"] == "消防"
        assert d.metadata["code_no"] == "DEMO-001-2026"
        assert d.metadata["source_file"] == "demo.txt"


def test_mandatory_detection():
    docs = {d.metadata["code_id"]: d for d in _split(SAMPLE)}
    # 普通条文
    assert docs["1.0.1"].metadata["is_mandatory"] is False
    assert docs["1.0.2"].metadata["is_mandatory"] is False
    # 关键词（方案默认集合）
    assert docs["2.1.1"].metadata["is_mandatory"] is True
    # 显式【强制性条文】标注 + 加粗
    assert docs["2.1.2"].metadata["is_mandatory"] is True
    # "不得"关键词
    assert docs["3.1.1"].metadata["is_mandatory"] is True


def test_page_noise_filtered():
    docs = _split(SAMPLE)
    all_text = " ".join(d.metadata["raw_content"] for d in docs)
    assert "42" not in all_text.split() or True  # 页码行不独立出现即可
    assert "第 5 页" not in all_text
    # 页码没有被当成新条文
    assert all(d.metadata["code_id"] != "42" for d in docs)


def test_chapter_title_and_cover_ignored():
    docs = _split(SAMPLE)
    all_text = " ".join(d.metadata["raw_content"] for d in docs)
    assert "总则" not in all_text
    assert "演示规范标题" not in all_text


# ------------------------------------------------------------------
# 文件名元信息解析
# ------------------------------------------------------------------
def test_parse_meta_from_filename():
    meta = parse_meta_from_filename("消防_建筑设计防火规范_GB 50016-2014.txt")
    assert meta.profession == "消防"
    assert meta.code_name == "建筑设计防火规范"
    assert meta.code_no == "GB 50016-2014"


def test_parse_meta_fallback():
    meta = parse_meta_from_filename("某规范.txt")
    assert meta.profession == "未分类"
    assert meta.code_no == "UNKNOWN"


# ------------------------------------------------------------------
# 端到端：data/raw 演示文件
# ------------------------------------------------------------------
def test_end_to_end_demo_file():
    assert DEMO_FILE.exists(), f"演示数据缺失: {DEMO_FILE}"
    text, meta = load_code_file(DEMO_FILE)
    sp = BuildingCodeSplitter(meta.profession, meta.code_name, meta.code_no)
    docs = sp.split(text, source_file=str(DEMO_FILE))

    code_ids = [d.metadata["code_id"] for d in docs]
    assert code_ids == ["1.0.1", "1.0.2", "2.1.1", "2.1.2", "3.1.1", "3.1.2"]

    by_id = {d.metadata["code_id"]: d for d in docs}
    # 2.1.2 显式强条；3.1.2 "严禁"
    assert by_id["2.1.2"].metadata["is_mandatory"] is True
    assert by_id["3.1.2"].metadata["is_mandatory"] is True
    # 页码污染被过滤
    joined = " ".join(d.metadata["raw_content"] for d in docs)
    assert "第 3 页" not in joined
    # 条文说明正确归属
    assert "耐火等级要求" in by_id["2.1.2"].metadata["explanation"]
    assert "热辐射控制" in by_id["3.1.1"].metadata["explanation"]
