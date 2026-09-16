"""ArchiRAG 配置管理。

严格保留项目开发方案中的全部配置项（QWEN_*/EMBEDDING_MODEL/FAISS_INDEX_PATH/
MYSQL_URL/REDIS_URL/WATCH_DIR 等），并扩展降级开关：
- dev 模式：MySQL/Redis/sentence-transformers/Qwen 缺失时自动降级并打印告警；
- prod 模式：关键依赖缺失直接拒绝启动。
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # ----- LLM：Qwen（DashScope，OpenAI 兼容模式）-----
    QWEN_API_KEY: str = ""
    QWEN_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    QWEN_MODEL: str = "qwen-plus"
    LLM_TEMPERATURE: float = 0.1

    # ----- Embedding -----
    EMBEDDING_MODEL: str = "BAAI/bge-large-zh-v1.5"
    EMBEDDING_DEVICE: str = "cpu"
    FALLBACK_EMBED_DIM: int = 1024

    # ----- FAISS -----
    FAISS_INDEX_PATH: str = "./data/faiss_index"
    TOP_K: int = 8

    # ----- MySQL（连不上时降级 SQLite）-----
    MYSQL_URL: str = "mysql+pymysql://root:root@localhost:3306/building_rag"
    SQLITE_PATH: str = "./data/building_rag.db"

    # ----- Redis（连不上时降级进程内 TTL 缓存）-----
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL: int = 3600

    # ----- 文件监控 -----
    WATCH_DIR: str = "./data/raw"

    # ----- 运行模式 -----
    APP_ENV: str = "dev"

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def is_prod(self) -> bool:
        return self.APP_ENV.lower() == "prod"

    @property
    def faiss_index_abs(self) -> Path:
        p = Path(self.FAISS_INDEX_PATH)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def watch_dir_abs(self) -> Path:
        p = Path(self.WATCH_DIR)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def sqlite_abs(self) -> Path:
        p = Path(self.SQLITE_PATH)
        return p if p.is_absolute() else PROJECT_ROOT / p


settings = Settings()
