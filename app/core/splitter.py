"""建筑规范结构化分块器 —— 本项目的核心模块。

建筑规范必须按"条文编号 + 条文正文 + 条文说明"分块，并保留
专业 / 规范名 / 规范号 / 是否强条等元数据，用于：
- 正文 + 说明拼接后向量化（语义检索）；
- code_id 元数据做条文号精确匹配（混合检索）；
- raw_content 供最终回答逐字引用原文。

支持的文本形态：
    3.2.1 厂房和仓库的耐火等级……          （新条文）
    条文说明：本条规定了……                 （上一条的条文说明）
    3.2.1 【条文说明】……                  （编号 + 说明同行）
    3.2.1 【强制性条文】厂房……**必须**……  （显式强条标注 / 加粗）
"""
from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document


class BuildingCodeSplitter:
    """建筑规范结构化分块器。

    Args:
        profession: 专业，如 建筑/结构/消防/电气
        code_name: 规范名称，如 建筑设计防火规范
        code_no: 规范编号，如 GB 50016-2014
        mandatory_keywords: 强条判定关键词（默认与项目方案一致，可配置收窄）
        strong_lines: 可选，PDF 黑体字行集合（未来从 pdfplumber 字体信息提取，
            命中即判定为强制性条文；这是真实强条的权威信号）
    """

    # 条文编号：1.0.1 / 3.2.10 / A.0.1 由 parser 层保证编号段为纯数字形态
    CODE_PATTERN = re.compile(r"^(\d+(?:\.\d+){1,3})\s+(.*)$")

    # 注意：必须要求出现"条文说明"四个字。
    # 项目方案示例中的正则各部分全部可选，会匹配任意行（含空串），
    # 导致普通正文被误切成 explanation。这里收紧为"必须含条文说明"。
    NOTE_PATTERN = re.compile(r"^(?:\d+(?:\.\d+){1,3}\s*)?【?\s*条\s*文\s*说\s*明\s*】?\s*[:：]?\s*(.*)$")

    # PDF 页眉页脚污染：纯数字页码 / "第 X 页"
    PAGE_NO_PATTERN = re.compile(r"^\d{1,4}$")
    PAGE_TAG_PATTERN = re.compile(r"^第\s*\d+\s*页$")

    # 显式强制性条文标注（权威信号，优先于关键词推断）
    STRONG_TAG_PATTERN = re.compile(r"【\s*(?:强制性条文|强条)\s*】")

    def __init__(
        self,
        profession: str,
        code_name: str,
        code_no: str,
        mandatory_keywords: list[str] | None = None,
        strong_lines: set[str] | None = None,
    ) -> None:
        self.profession = profession
        self.code_name = code_name
        self.code_no = code_no
        # 默认与项目开发方案保持一致：必须 / 严禁 / 应符合 / 不得
        self.mandatory_keywords = mandatory_keywords or ["必须", "严禁", "应符合", "不得"]
        self.strong_lines = strong_lines or set()

    # ------------------------------------------------------------------
    # 公开入口
    # ------------------------------------------------------------------
    def split(self, raw_text: str, source_file: str) -> list[Document]:
        documents: list[Document] = []
        current: dict[str, Any] | None = None
        mode = "content"  # content / explanation

        for raw_line in raw_text.split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            # 过滤 PDF 页码 / 页眉页脚
            if self.PAGE_NO_PATTERN.match(line) or self.PAGE_TAG_PATTERN.match(line):
                continue

            # 1) 条文说明（优先于条文号判定，因为可能是 "3.2.1 条文说明：…"）
            note = self.NOTE_PATTERN.match(line)
            # 必须真的含"条文说明"四字，否则视为普通行
            if note and ("条文说明" in line):
                mode = "explanation"
                rest = note.group(1).strip()
                if current is None:
                    # 没有归属条文的孤立说明，直接忽略（通常是目录残留）
                    continue
                if rest:
                    current["explanation"] = rest
                continue

            # 2) 新条文开始
            m = self.CODE_PATTERN.match(line)
            if m:
                if current is not None:
                    documents.append(self._build_doc(current, source_file))
                code_id, content = m.group(1), m.group(2)
                current = {
                    "code_id": code_id,
                    "content": content,
                    "explanation": "",
                    "is_mandatory": self._detect_mandatory(line, content),
                }
                mode = "content"
                continue

            # 3) 续接行
            if current is not None:
                if mode == "content":
                    current["content"] += " " + line
                else:
                    current["explanation"] += " " + line
                # 续接行也可能携带强条信号（加粗符号、黑体行）
                if not current["is_mandatory"]:
                    current["is_mandatory"] = self._detect_mandatory(line, line)
            # current is None 时：章标题 / 目录 / 封面信息等游离行，忽略

        if current is not None:
            documents.append(self._build_doc(current, source_file))
        return documents

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    def _detect_mandatory(self, line: str, content: str) -> bool:
        """检测强制性条文。

        判定信号（按权威度排序）：
        1. 显式标注【强制性条文】/【强条】；
        2. PDF 黑体字行（strong_lines，未来接入）；
        3. Markdown 加粗符号 **（文本化处理中间态）；
        4. 关键词推断（项目方案默认：必须/严禁/应符合/不得）。

        说明：真实规范中强条以黑体字专门排版，关键词法会有误判，
        生产环境应通过 PDF 字体信息构造 strong_lines 覆盖关键词结果。
        """
        if self.STRONG_TAG_PATTERN.search(line):
            return True
        if line.strip() in self.strong_lines:
            return True
        if "**" in line:
            return True
        return any(k in content for k in self.mandatory_keywords)

    def _build_doc(self, item: dict[str, Any], source_file: str) -> Document:
        # 拼接用于向量化的文本（正文 + 说明一起做语义检索）
        explanation = item["explanation"] or "无"
        text_for_embedding = (
            f"《{self.code_name}》{self.code_no} 第{item['code_id']}条：\n"
            f"{item['content']}\n"
            f"条文说明：{explanation}"
        )
        return Document(
            page_content=text_for_embedding,
            metadata={
                "profession": self.profession,
                "code_name": self.code_name,
                "code_no": self.code_no,
                "code_id": item["code_id"],  # 关键：用于条文号精确匹配
                "is_mandatory": item["is_mandatory"],
                "source_file": source_file,
                "raw_content": item["content"],  # 用于最终回答引用原文
                "explanation": item["explanation"],
            },
        )
