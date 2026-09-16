"""规范文档解析层。

职责：把磁盘上的规范文件（txt / docx / pdf）读取为纯文本，并从文件名
约定中解析规范元信息（专业 / 规范名 / 规范号）。

文件名约定：``{专业}_{规范名}_{规范号}.{ext}``
例如：``消防_建筑设计防火规范_GB 50016-2014.txt``

注意：真正的"条文级结构化分块"在 splitter.py 中完成，本模块只负责
"文件 -> 原始文本 + 元信息"。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CodeMeta:
    """一本规范的元信息。"""

    profession: str  # 建筑/结构/消防/电气
    code_name: str   # 《建筑设计防火规范》
    code_no: str     # GB 50016-2014


def parse_meta_from_filename(path: str | Path) -> CodeMeta:
    """按 ``专业_规范名_规范号`` 约定解析文件名。

    无法解析时给出兜底值，不抛异常（监控入库场景不能因命名不规范而中断）。
    """
    stem = Path(path).stem
    parts = stem.split("_")
    if len(parts) >= 3:
        profession, code_name, code_no = parts[0], parts[1], "_".join(parts[2:])
    elif len(parts) == 2:
        profession, code_name, code_no = "未分类", parts[0], parts[1]
    else:
        profession, code_name, code_no = "未分类", stem, "UNKNOWN"
        logger.warning("文件名不符合『专业_规范名_规范号』约定，使用兜底元信息: %s", stem)
    return CodeMeta(profession=profession, code_name=code_name, code_no=code_no)


def _read_txt(path: Path) -> str:
    # 规范文本常见编码依次尝试
    for enc in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"无法识别文件编码: {path}")


def _read_docx(path: Path) -> str:
    try:
        import docx  # python-docx
    except ImportError as e:
        raise RuntimeError(
            "解析 .docx 需要 python-docx，请执行: pip install python-docx"
        ) from e
    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)


def _read_pdf(path: Path) -> str:
    try:
        import pdfplumber
    except ImportError as e:
        raise RuntimeError(
            "解析 .pdf 需要 pdfplumber，请执行: pip install pdfplumber"
        ) from e
    pages: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)


_READERS = {
    ".txt": _read_txt,
    ".docx": _read_docx,
    ".doc": _read_docx,
    ".pdf": _read_pdf,
}


def read_raw_text(path: str | Path) -> str:
    """读取规范文件为纯文本。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"规范文件不存在: {p}")
    reader = _READERS.get(p.suffix.lower())
    if reader is None:
        raise ValueError(f"不支持的规范文件类型: {p.suffix}（支持 txt/docx/pdf）")
    return reader(p)


def load_code_file(path: str | Path) -> tuple[str, CodeMeta]:
    """一步完成：读取文本 + 解析元信息。"""
    p = Path(path)
    meta = parse_meta_from_filename(p)
    text = read_raw_text(p)
    return text, meta
