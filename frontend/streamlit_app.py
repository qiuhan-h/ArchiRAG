"""ArchiRAG Streamlit 前端 —— DeepSeek 风格聊天界面。

页面：
- 对话问答（聊天式 + SSE 流式输出 + 左侧会话栏）
- 合规审查
- 统计看板

启动：
    streamlit run frontend/streamlit_app.py
"""
"""
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
import streamlit as st

# 确保 app 包可导入
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 本地裸跑默认 localhost:8000；容器内通过环境变量指向 backend 服务名
API_BASE = os.environ.get("ARCHIRAG_API_BASE", "http://localhost:8000/api")
PROFESSIONS = ["", "消防", "电气", "结构", "暖通"]

st.set_page_config(
    page_title="ArchiRAG 建筑规范智能问答",
    page_icon="📋",
    layout="wide",
)

# ------------------------------------------------------------------
# 最小侵入 CSS：只调间距/圆角/侧栏宽度，不替换任何组件
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] { width: 280px !important; }
    .block-container { padding-top: 1.5rem; max-width: 900px; }
    .archi-ref { font-size: 0.88rem; color: #555; }
    .archi-tag {
        display:inline-block; padding:0 6px; border-radius:4px;
        background:#ffe8e8; color:#d4380d; font-size:0.78rem; margin-right:6px;
    }
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# API 封装
# ------------------------------------------------------------------
def api_health() -> dict:
    try:
        return requests.get(f"{API_BASE}/health", timeout=2).json()
    except Exception:
        return {}


def api_list_sessions() -> list[dict]:
    try:
        return requests.get(f"{API_BASE}/sessions", timeout=5).json()
    except Exception:
        return []


def api_get_messages(session_id: int) -> list[dict]:
    try:
        return requests.get(f"{API_BASE}/sessions/{session_id}/messages", timeout=5).json()
    except Exception:
        return []


def api_delete_session(session_id: int) -> bool:
    try:
        r = requests.delete(f"{API_BASE}/sessions/{session_id}", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def stream_answer(
    question: str,
    profession: str | None,
    only_mandatory: bool,
    session_id: int | None,
):
    """调用 SSE 流式接口，返回 (meta, token_generator)。"""
    resp = requests.post(
        f"{API_BASE}/qa/stream",
        json={
            "question": question,
            "profession": profession,
            "only_mandatory": only_mandatory,
            "session_id": session_id,
        },
        stream=True,
        timeout=(10, 120),
    )
    resp.raise_for_status()
    lines = resp.iter_lines(decode_unicode=True)

    # 读取第一条事件（必为 meta）
    meta = None
    for raw in lines:
        if raw:
            meta = json.loads(raw.removeprefix("data: "))
            break
    if meta is None:
        raise RuntimeError("流式响应为空")

    def token_gen():
        if meta.get("preset_answer"):
            yield meta["preset_answer"]
            return
        for raw in lines:
            if not raw:
                continue
            ev = json.loads(raw.removeprefix("data: "))
            if ev.get("type") == "token":
                yield ev["delta"]
            elif ev.get("type") == "done":
                return

    return meta, token_gen


# ------------------------------------------------------------------
# session_state 初始化（只初始化一次，不覆盖用户状态）
# ------------------------------------------------------------------
def init_state() -> None:
    defaults = {
        "page": "chat",
        "session_id": None,
        "messages": [],
        "sessions": [],
        "profession": "",
        "only_mandatory": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ------------------------------------------------------------------
# 渲染辅助
# ------------------------------------------------------------------
def render_references(refs: list[dict]) -> None:
    """在 assistant 气泡内渲染引用条文卡片。"""
    if not refs:
        return
    st.markdown("**引用条文：**")
    for ref in refs:
        tag = '<span class="archi-tag">强制性条文</span>' if ref.get("is_mandatory") else ""
        # code_no 可能是导入时由 code_name 推导的批次号（同名或其细化形式），
        # 此时不再重复展示；只有真实规范号（如 GB 50016-2014）才显示
        code_no = ref.get("code_no", "")
        code_name = ref.get("code_name", "")
        show_code_no = bool(code_no) and not (
            code_no == code_name or code_no.startswith(code_name)
        )
        no_part = f"{code_no} " if show_code_no else ""
        with st.container(border=True):
            st.markdown(
                f'{tag}《{code_name}》{no_part}'
                f'**第{ref["code_id"]}条**',
                unsafe_allow_html=True,
            )
            st.caption(ref.get("content", ""))


def render_history() -> None:
    """重放当前会话的历史消息。"""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_references(msg.get("references", []))


def group_sessions(sessions: list[dict]) -> dict[str, list[dict]]:
    """按 今天/昨天/7天内/更早 分组（北京时间）。"""
    groups: dict[str, list[dict]] = {"今天": [], "昨天": [], "7天内": [], "更早": []}
    bj_tz = timezone(timedelta(hours=8))
    now_bj = datetime.now(bj_tz)
    today = now_bj.date()
    for s in sessions:
        try:
            ts = datetime.fromisoformat(s["updated_at"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except Exception:
            groups["更早"].append(s)
            continue
        delta = (now_bj - ts.astimezone(bj_tz)).days
        local_date = ts.astimezone(bj_tz).date()
        if local_date == today:
            groups["今天"].append(s)
        elif local_date == today - timedelta(days=1):
            groups["昨天"].append(s)
        elif delta < 7:
            groups["7天内"].append(s)
        else:
            groups["更早"].append(s)
    return groups


def switch_session(sid: int) -> None:
    st.session_state.session_id = sid
    st.session_state.messages = api_get_messages(sid)
    st.session_state.page = "chat"


def new_chat() -> None:
    st.session_state.session_id = None
    st.session_state.messages = []
    st.session_state.page = "chat"


# ------------------------------------------------------------------
# 侧边栏
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📋 ArchiRAG")
    if st.button("＋ 新建对话", use_container_width=True, type="primary"):
        new_chat()
        st.rerun()

    st.divider()

    # 页面切换（原生 radio，状态稳定）
    page = st.radio(
        "功能",
        options=["chat", "audit", "stats"],
        format_func=lambda x: {"chat": "💬 对话问答", "audit": "✅ 合规审查", "stats": "📊 统计看板"}[x],
        label_visibility="collapsed",
    )
    st.session_state.page = page

    if page == "chat":
        # 对话选项
        with st.expander("检索选项", expanded=False):
            st.session_state.profession = st.selectbox(
                "专业", PROFESSIONS,
                index=PROFESSIONS.index(st.session_state.profession),
            )
            st.session_state.only_mandatory = st.checkbox(
                "只检索强制性条文", value=st.session_state.only_mandatory
            )

        st.divider()
        # 会话列表
        st.session_state.sessions = api_list_sessions()
        groups = group_sessions(st.session_state.sessions)
        for label, items in groups.items():
            if not items:
                continue
            st.caption(label)
            for s in items:
                c1, c2 = st.columns([5, 1])
                is_current = st.session_state.session_id == s["id"]
                if c1.button(
                    s["title"][:18],
                    key=f"sess_{s['id']}",
                    use_container_width=True,
                    type="primary" if is_current else "secondary",
                ):
                    switch_session(s["id"])
                    st.rerun()
                if c2.button("🗑", key=f"del_{s['id']}", help="删除会话"):
                    api_delete_session(s["id"])
                    if is_current:
                        new_chat()
                    st.rerun()

    # 系统状态
    with st.expander("系统状态"):
        h = api_health()
        if h:
            st.caption(f"数据库: {h.get('db_type', '?')}")
            st.caption(f"缓存: {h.get('cache_type', '?')}")
            st.caption(f"LLM: {h.get('llm_mode', '?')}")
            st.caption(f"Embedding: {'降级' if h.get('embedding_fallback') else 'bge'}")
        else:
            st.warning("后端未启动，请运行: uvicorn app.main:app --port 8000")


# ------------------------------------------------------------------
# 页面：对话问答
# ------------------------------------------------------------------
if st.session_state.page == "chat":
    st.title("建筑规范智能问答")

    # 历史消息
    render_history()

    # 空会话欢迎语 + 示例问题（原生按钮）
    if not st.session_state.messages:
        st.info("基于建筑规范库的智能问答，回答附带条文引用，可追溯、不杜撰。")
        examples = [
            "甲类厂房与重要公共建筑的防火间距是多少？",
            "重要公共建筑的耐火等级有什么要求？",
            "第3.1.2条是怎么规定的？",
        ]
        cols = st.columns(len(examples))
        for col, ex in zip(cols, examples):
            if col.button(ex, key=f"ex_{ex}", use_container_width=True):
                st.session_state["_pending_question"] = ex
                st.rerun()

    # 输入框
    prompt = st.chat_input("输入你的建筑规范问题…")
    pending = st.session_state.pop("_pending_question", None)
    question = prompt or pending

    if question:
        prof = st.session_state.profession or None
        sid = st.session_state.session_id

        # 先显示用户气泡
        with st.chat_message("user"):
            st.markdown(question)

        try:
            meta, token_gen = stream_answer(
                question, prof, st.session_state.only_mandatory, sid
            )
        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"请求失败: {e}")
            st.stop()

        # 流式渲染 assistant 回答
        with st.chat_message("assistant"):
            answer = st.write_stream(token_gen)
            refs = meta.get("references", [])
            render_references(refs)
            flag_parts = []
            if meta.get("cache_hit"):
                flag_parts.append("缓存命中")
            if meta.get("llm_degraded"):
                flag_parts.append("降级模式")
            if flag_parts:
                st.caption(" ｜ ".join(flag_parts))

        # 更新本地会话状态
        st.session_state.session_id = meta.get("session_id", sid)
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "references": refs}
        )
        # 刷新左侧会话列表
        st.session_state.sessions = api_list_sessions()


# ------------------------------------------------------------------
# 页面：合规审查
# ------------------------------------------------------------------
elif st.session_state.page == "audit":
    st.title("合规审查")
    audit_text = st.text_area(
        "审查内容", height=180, placeholder="粘贴设计说明/施工方案片段…"
    )
    audit_prof = st.selectbox("限定专业", PROFESSIONS, key="audit_prof_box")
    prof2 = audit_prof if audit_prof else None

    if st.button("开始审查", type="primary") and audit_text:
        with st.spinner("审查中…"):
            try:
                resp = requests.post(
                    f"{API_BASE}/audit",
                    json={"audit_text": audit_text, "profession": prof2},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                st.error(f"请求失败: {e}")
                data = None

        if data:
            if data["compliance"] == "pass":
                st.success("✅ 审查通过：未命中强制性条文")
            else:
                st.error(f"❌ 需关注：命中 {data['hit_mandatory_count']} 条强制性条文")
            for v in data.get("violations", []):
                with st.container(border=True):
                    st.markdown(f"**{v['code_name']} 第{v['code_id']}条**")
                    st.caption(f"审查句子: {v['sentence']}")
                    st.caption(f"强条原文: {v['mandatory_clause'][:120]}")


# ------------------------------------------------------------------
# 页面：统计看板
# ------------------------------------------------------------------
elif st.session_state.page == "stats":
    st.title("统计看板")
    try:
        stats = requests.get(f"{API_BASE}/stats", timeout=5).json()
    except Exception as e:
        st.error(f"获取统计失败: {e}")
        stats = {"by_profession": [], "query_stats": {}}

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**按专业统计**")
        prof_data = stats.get("by_profession", [])
        if prof_data:
            import pandas as pd
            st.dataframe(pd.DataFrame(prof_data), use_container_width=True)
        else:
            st.info("暂无条文数据")

    with col_b:
        st.markdown("**查询统计**")
        qs = stats.get("query_stats", {})
        m1, m2 = st.columns(2)
        m1.metric("总查询数", qs.get("total_queries", 0))
        m2.metric("缓存命中", qs.get("cache_hits", 0))
        m3, m4 = st.columns(2)
        m3.metric("命中率", f"{qs.get('cache_hit_rate', 0):.1%}")
        m4.metric("平均耗时", f"{qs.get('avg_latency_ms', 0)}ms")
        st.metric("降级次数", qs.get("degraded_count", 0))

    st.divider()
    st.markdown("**已入库规范条文**")
    try:
        docs = requests.get(f"{API_BASE}/documents?limit=50", timeout=5).json()
        if docs["items"]:
            import pandas as pd
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "规范号": d["code_no"],
                            "规范名": d["code_name"],
                            "条文号": d["code_id"],
                            "专业": d["profession"],
                            "强条": "是" if d["is_mandatory"] else "",
                        }
                        for d in docs["items"]
                    ]
                ),
                use_container_width=True,
            )
        else:
            st.info("暂无入库条文")
    except Exception as e:
        st.error(f"获取文档列表失败: {e}")
