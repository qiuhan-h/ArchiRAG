"""端到端冒烟：入库→问答→写DB→缓存→统计。"""
import time
import json

from app.core.qa_service import QAService
from app.core.cache import cache_get, cache_set, make_cache_key, cache_type
from app.db import crud
from app.db.session import db_type

svc = QAService()
q = "甲类厂房与重要公共建筑的防火间距是多少"

# 第一次问答（未命中缓存）
t0 = time.time()
r1 = svc.ask(q)
ms1 = int((time.time() - t0) * 1000)
key = make_cache_key(q)
cache_set(key, json.dumps(r1.to_dict()), ttl=60)
print("[第一次] %dms | cache=%s | db=%s" % (ms1, cache_type(), db_type()))
print("  answer: %s..." % r1.answer[:40])

# 第二次（缓存命中）
t0 = time.time()
cached = cache_get(key)
ms2 = int((time.time() - t0) * 1000)
print("[缓存]   %dms | hit=%s" % (ms2, cached is not None))

# 写入 DB 日志
crud.log_query(q, r1.answer, r1.references, r1.llm_mode, r1.llm_degraded, True, ms1)
stats = crud.query_stats()
print("[DB]     queries=%d cache_hits=%d" % (stats["total_queries"], stats["cache_hits"]))
print("[统计]   avg=%sms degraded=%d" % (stats["avg_latency_ms"], stats["degraded_count"]))
