# ArchiRAG —— 建筑规范智能问答与合规审查系统

基于 RAG 架构的建筑规范知识引擎：规范文件（PDF/Word/TXT）→ 条文级结构化分块 →
向量化入库（FAISS）→ 语义 + 条文号混合检索（BM25）→ Qwen 生成"逐条引用原文"的回答；
并支持设计说明对强制性条文的遗漏审查。

## 核心特性

- **三路混合检索**：FAISS 语义向量 + BM25 真倒排 + 条文号正则精确通道，RRF 融合排序
- **SSE 流式输出**：`POST /api/qa/stream`，Qwen 真流式（`stream=True` 逐 token）；
  无 API Key 时降级为抽取式回答分块输出，前端打字机效果一致
- **聊天式前端**：DeepSeek 风格会话界面，多轮对话气泡 + 引用条文卡片；
  会话与消息持久化（SQLite/MySQL），左侧按"今天/昨天/7天内"分组，可切换、删除
- **引用约束生成**：回答必须逐字引用条文原文并标注规范号/条文号，强条红色标记，
  检索为空时明确回答"未找到"，不杜撰
- **合规审查**：设计说明分句 → 强制性条文匹配 → pass/fail 判定与违规清单
- **全链路降级**：LLM / Embedding / MySQL / Redis 缺失时自动降级，零外部服务可跑通
- **增量同步**：watchdog 独立进程监控规范目录，文件 hash 去重，避免多 worker 重复入库
- **LLM Query 改写**：Qwen 把口语问题转为规范术语 + 同义词扩展，双轮检索 RRF 融合最大化召回

## 技术栈（严格按项目开发方案）

Python 3.11+ · LangChain · FastAPI · MySQL · FAISS · Redis · Qwen(DashScope) ·
BAAI/bge-large-zh-v1.5 · Streamlit · Docker

## 降级策略（开发环境零外部服务可跑通）

| 组件 | 生产模式 | 缺失时降级 |
|------|---------|-----------|
| LLM | Qwen（OpenAI 兼容协议） | 本地抽取式 FakeLLM（基于检索条文组织回答） |
| Embedding | bge-large-zh-v1.5（sentence-transformers） | 确定性字符 n-gram 向量（无 torch，可跑通链路） |
| MySQL | MySQL 8.0 | SQLite 本地文件 |
| Redis | Redis 7 | 进程内 TTL 缓存 |
| 规范解析 | pdfplumber / python-docx | txt 直读；缺包时给出安装提示 |

> 真实语义检索效果需安装重依赖：`pip install -r requirements-embed.txt`

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env       # 填入 QWEN_API_KEY 后为真实模式；留空即降级模式
pytest                        # 全量测试（76 个）
```

启动服务：

```powershell
# 入库演示规范（可选）
python scripts/ingest.py data/raw

# 后端 + 前端
uvicorn app.main:app --port 8000
streamlit run frontend/streamlit_app.py
```

## API 一览

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/qa` | 一次性问答（含缓存 + 查询日志） |
| POST | `/api/qa/stream` | **SSE 流式问答**（meta → token* → done，自动建/续会话） |
| POST | `/api/audit` | 合规审查 |
| GET | `/api/stats` | 专业统计 + 查询命中率/耗时统计 |
| GET/POST/DELETE | `/api/documents` | 条文列表 / 入库 / 删除（FAISS + DB 双写） |
| GET/POST/DELETE | `/api/sessions` | 会话列表 / 新建 / 删除（含 `/{id}/messages` 历史） |

## 目录结构

```
ArchiRAG/
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI 入口（lifespan + CORS + 健康检查）
│   ├── config.py                   # Settings 配置（含降级开关 + 路径计算）
│   ├── api/
│   │   ├── __init__.py
│   │   ├── qa.py                   # /api/qa + /api/qa/stream（SSE 流式 + 缓存 + 会话）
│   │   ├── sessions.py             # /api/sessions（会话 CRUD + 消息历史）
│   │   ├── audit.py                # /api/audit（合规审查）
│   │   ├── stats.py                # /api/stats（专业统计 + 查询统计）
│   │   └── documents.py           # /api/documents（列表/入库/删除，FAISS+DB 双写）
│   ├── core/
│   │   ├── __init__.py
│   │   ├── parser.py              # txt/docx/pdf 解析 + 文件名元信息解析
│   │   ├── splitter.py            # 条文级结构化分块（编号+正文+说明+强条检测）
│   │   ├── embedder.py            # bge 语义向量 / 哈希降级 + 词面噪声闸
│   │   ├── vector_store.py        # FAISS IndexFlatIP + 中文路径序列化 + 元数据过滤
│   │   ├── hybrid_retriever.py   # 语义+BM25+条文号三路 RRF 融合 + Query 改写双轮召回
│   │   ├── query_rewriter.py     # LLM Query 改写（口语→规范术语，temperature=0）
│   │   ├── prompt.py              # 引用约束 System Prompt（五条强制规则）
│   │   ├── llm.py                 # Qwen（OpenAI 兼容）+ chat_stream 流式 + 抽取式降级
│   │   ├── qa_service.py         # 问答主链路（ask / ask_stream，零命中/降级三分支）
│   │   ├── audit_service.py      # 合规审查（分句→强条匹配→pass/fail 判定）
│   │   └── cache.py              # Redis / 进程内 TTL 降级（线程安全 + 惰性过期）
│   ├── db/
│   │   ├── __init__.py
│   │   ├── models.py             # 五表：CodeDocument/QueryLog/AuditRecord/ChatSession/ChatMessage
│   │   ├── session.py            # 引擎单例：MySQL pool_pre_ping → 失败降级 SQLite
│   │   └── crud.py               # CRUD + stats_by_profession + query_stats
│   ├── watcher/
│   │   ├── __init__.py
│   │   └── file_monitor.py       # watchdog 独立进程：全量扫描+增量事件+hash去重+防抖
│   └── schemas/
│       ├── __init__.py
│       └── models.py             # Pydantic 请求/响应模型（QA/Audit/Stats/Docs/Sessions）
├── frontend/
│   └── streamlit_app.py          # DeepSeek 风格：会话栏+消息气泡+打字机+引用卡片+统计看板
├── data/
│   ├── raw/                      # 原始规范文件（命名：专业_规范名_规范号.txt）
│   ├── processed/                # 结构化 JSON（预留）
│   ├── faiss_index/              # FAISS 持久化（index.faiss + index.pkl）
│   ├── seed/
│   │   └── shu_ju.jsonl          # 数据源（400 条四专业规范条文）
│   ├── examples/
│   │   └── 消防_演示规范_DEMO-001-2026.txt  # 演示规范
│   ├── modelscope/
│   │   └── bge-large-zh-v1.5/    # bge 模型权重（1.24GB，从 ModelScope 下载）
│   └── test_questions.txt        # 60 条前端测试集
├── docker/
│   ├── Dockerfile                # python:3.11-slim + faiss/pdf 系统依赖 + 可选 bge
│   ├── docker-compose.yml        # 五服务：backend/watcher/frontend/db/mysql/redis + nginx(prod)
│   ├── .dockerignore
│   ├── DEPLOY.md                 # 生产部署指南（SSL/域名/安全加固）
│   └── nginx/
│       ├── Dockerfile            # nginx:alpine
│       └── nginx.conf            # HTTPS+WebSocket+SSE 反代配置
├── tests/
│   ├── conftest.py               # 公共夹具（强制 dev 模式 + 独立 FAISS 临时目录）
│   ├── test_splitter.py          # M1 条文分块测试
│   ├── test_retrieval.py         # M2 混合检索测试
│   ├── test_qa.py                # M3 问答链路测试
│   ├── test_db.py                # M4 DB CRUD + 缓存测试
│   ├── test_api.py               # M5 API 端到端测试
│   ├── test_stream_chat.py       # M7 流式问答 + 会话测试
│   ├── test_watcher_eval.py      # M6 watcher 去重/全量扫描 + 评估脚本测试
│   └── test_import_jsonl.py      # JSONL 导入测试
├── scripts/
│   ├── __init__.py
│   ├── ingest.py                 # 规范入库 CLI（txt 整本→splitter→FAISS+DB）
│   ├── import_jsonl.py           # JSONL 结构化导入（坏行容错+幂等双写）
│   ├── evaluate.py               # 评估脚本（recall@k + 关键词命中 + 准确率）
│   └── smoke_e2e.py              # 端到端冒烟测试
├── .env                          # 环境变量（API Key/DB/Redis 路径）
├── .env.example                  # 环境变量模板
├── .env.prod.example             # 生产环境变量模板
├── .gitignore
├── .dockerignore
├── requirements.txt              # 核心依赖（无 torch）
├── requirements-embed.txt        # 可选重依赖（sentence-transformers + bge）
├── pytest.ini
└── README.md
```

## 开发里程碑

- [x] M1 骨架 + 配置降级 + 条文级结构化分块
- [x] M2 Embedding + FAISS + 真实 BM25 混合检索
- [x] M3 Qwen LLM + 引用约束 Prompt + 问答链路
- [x] M4 MySQL 模型/CRUD + Redis 缓存（均含降级）
- [x] M5 FastAPI 接口（qa/audit/stats/documents）+ Streamlit 三页前端
- [x] M6 文件增量同步（独立进程）+ Docker + 评估脚本（KPI 全部达标）
- [x] M7 SSE 流式问答 + 会话持久化 + DeepSeek 风格聊天界面改造
- [x] M8 LLM Query 改写 + Docker 生产部署（Nginx/HTTPS/资源限制/bge 真实模型）
