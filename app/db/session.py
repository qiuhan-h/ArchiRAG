"""数据库 Session 管理（严格按方案：MySQL；连不上降级 SQLite）。

- 首次调用 get_session() 时尝试连接 MySQL（settings.MYSQL_URL）；
- 连接失败（pymysql 缺失/MySQL 未启动/权限拒绝）时自动降级到
  SQLite（settings.SQLITE_PATH），并建全部表，打印告警；
- prod 模式下 MySQL 连不上直接拒绝启动。
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.db.models import Base

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None
_db_type: str = ""  # "mysql" / "sqlite"


def _try_mysql() -> Engine:
    """尝试创建 MySQL 引擎并验证连接。"""
    url = settings.MYSQL_URL
    engine = create_engine(
        url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        pool_recycle=3600,
        echo=False,
    )
    # 验证连接（from ... import 避免顶层依赖）
    from sqlalchemy import text

    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return engine


def _make_sqlite_engine() -> Engine:
    """创建 SQLite 引擎（降级路径）。"""
    db_path = settings.sqlite_abs
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Windows 中文路径用 4 个斜杠绕过 sqlite3 连接字符串解析
    engine = create_engine(
        f"sqlite:///{db_path.as_posix()}",
        echo=False,
        connect_args={"check_same_thread": False},
    )
    return engine


def _ensure_engine() -> Engine:
    """惰性初始化引擎：MySQL → 失败降级 SQLite。"""
    global _engine, _SessionLocal, _db_type
    if _engine is not None:
        return _engine

    # 1. 尝试 MySQL
    try:
        _engine = _try_mysql()
        _db_type = "mysql"
        logger.info("数据库连接成功: MySQL (%s)", settings.MYSQL_URL.split("@")[-1])
    except Exception as e:
        if settings.is_prod:
            logger.error("prod 模式下 MySQL 不可用，拒绝启动: %s", e)
            raise
        logger.warning(
            "MySQL 连接失败（%s），降级到 SQLite: %s",
            type(e).__name__,
            settings.sqlite_abs,
        )
        _engine = _make_sqlite_engine()
        _db_type = "sqlite"

    # 2. 建表
    Base.metadata.create_all(_engine)

    # 3. Session 工厂
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def get_engine() -> Engine:
    """获取当前引擎（供需要直接操作引擎的场景使用）。"""
    return _ensure_engine()


def get_session() -> Session:
    """获取一个数据库 Session。

    调用方负责 close（通常用 get_session_ctx 的上下文管理器）。
    """
    _ensure_engine()
    return _SessionLocal()  # type: ignore[union-attr]


@contextmanager
def get_session_ctx() -> Generator[Session, None, None]:
    """上下文管理器：自动 close + rollback on error。"""
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def db_type() -> str:
    """当前数据库类型：'mysql' / 'sqlite'。"""
    if not _db_type:
        _ensure_engine()
    return _db_type
