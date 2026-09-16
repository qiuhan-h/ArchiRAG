"""缓存层（严格按方案：Redis；连不上降级进程内 TTL）。

- 真实模式：redis-py 连接 settings.REDIS_URL，支撑缓存命中 0.2s KPI；
- 降级模式：进程内 dict + TTL，过期自动清理，单进程内有效；
- 两者接口一致：get(key) / set(key, value, ttl) / delete(key) / exists(key)。
"""
from __future__ import annotations

import logging
import time
import threading
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 进程内 TTL 缓存（降级用）
# ------------------------------------------------------------------
class InMemoryTTLCache:
    """进程内带过期时间的简易缓存。"""

    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float]] = {}  # key -> (value, expire_ts)
        self._lock = threading.Lock()

    def _cleanup(self) -> None:
        """清理已过期 key（惰性 + 主动扫描）。"""
        now = time.time()
        expired = [k for k, (_, ts) in self._store.items() if ts <= now]
        for k in expired:
            self._store.pop(k, None)

    def get(self, key: str) -> Any | None:
        with self._lock:
            self._cleanup()
            item = self._store.get(key)
            if item is None:
                return None
            value, expire_ts = item
            if expire_ts <= time.time():
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        with self._lock:
            self._store[key] = (
                value,
                time.time() + (ttl if ttl is not None else settings.CACHE_TTL),
            )
            return True

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._store.pop(key, None) is not None

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def size(self) -> int:
        with self._lock:
            self._cleanup()
            return len(self._store)


# ------------------------------------------------------------------
# Redis 封装
# ------------------------------------------------------------------
class RedisCache:
    """Redis 缓存（真实模式）。"""

    def __init__(self) -> None:
        import redis as redis_lib

        self._client = redis_lib.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        # 验证连接
        self._client.ping()

    def get(self, key: str) -> Any | None:
        return self._client.get(key)

    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        return self._client.setex(
            key, ttl if ttl is not None else settings.CACHE_TTL, value
        )

    def delete(self, key: str) -> bool:
        return bool(self._client.delete(key))

    def exists(self, key: str) -> bool:
        return bool(self._client.exists(key))

    def size(self) -> int:
        return int(self._client.dbsize())


# ------------------------------------------------------------------
# 单例工厂
# ------------------------------------------------------------------
_cache: RedisCache | InMemoryTTLCache | None = None
_cache_type: str = ""  # "redis" / "memory"


def _ensure_cache() -> RedisCache | InMemoryTTLCache:
    global _cache, _cache_type
    if _cache is not None:
        return _cache

    # 尝试 Redis
    try:
        _cache = RedisCache()
        _cache_type = "redis"
        logger.info("Redis 缓存连接成功: %s", settings.REDIS_URL)
    except Exception as e:
        if settings.is_prod:
            logger.error("prod 模式下 Redis 不可用，拒绝启动: %s", e)
            raise
        logger.warning(
            "Redis 连接失败（%s），降级到进程内 TTL 缓存", type(e).__name__
        )
        _cache = InMemoryTTLCache()
        _cache_type = "memory"
    return _cache


def cache_type() -> str:
    """当前缓存类型：'redis' / 'memory'。"""
    if not _cache_type:
        _ensure_cache()
    return _cache_type


def get_cache() -> RedisCache | InMemoryTTLCache:
    """获取缓存单例。"""
    return _ensure_cache()


# ------------------------------------------------------------------
# 便捷函数（直接调用，面向 API 层使用）
# ------------------------------------------------------------------
def cache_get(key: str) -> Any | None:
    return get_cache().get(key)


def cache_set(key: str, value: Any, ttl: int | None = None) -> bool:
    return get_cache().set(key, value, ttl)


def cache_delete(key: str) -> bool:
    return get_cache().delete(key)


def make_cache_key(question: str, profession: str | None = None, k: int | None = None) -> str:
    """构造缓存键（问答结果缓存用）。"""
    parts = [question.strip()]
    if profession:
        parts.append(f"p:{profession}")
    if k is not None:
        parts.append(f"k:{k}")
    import hashlib

    return f"qa:{hashlib.md5('|'.join(parts).encode('utf-8')).hexdigest()}"
