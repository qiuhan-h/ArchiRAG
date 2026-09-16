"""ArchiRAG 一键启动脚本（后端 + watcher + 前端 + 首次入库）。

用法：
    python run_backend.py                # 启动全部
    python run_backend.py --skip-ingest   # 跳过首次入库检查
    python run_backend.py --skip-watcher  # 不启动文件监控
    python run_backend.py --skip-frontend  # 不启动 Streamlit 前端
    python run_backend.py --backend-port 8001  # 自定义后端端口

启动顺序：
    1. 检查 FAISS 索引是否存在，若不存在且 data/raw 有文件 → 首次入库
    2. 启动 watcher（独立进程，监控 data/raw 增量入库）
    3. 启动 uvicorn 后端 API
    4. 启动 streamlit 前端
    5. Ctrl+C 优雅停止全部子进程
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# 全局子进程表：[(name, Popen, Thread)]
PROCESSES: list[tuple[str, subprocess.Popen, threading.Thread]] = []


def _env_with_utf8() -> dict[str, str]:
    """子进程环境变量：继承当前环境 + 强制 Python 子进程使用 utf-8 输出。"""
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _stream_output(proc: subprocess.Popen, name: str) -> None:
    """转发子进程 stdout 行，每行加 [name] 前缀。"""
    if proc.stdout is None:
        return
    try:
        for line in iter(proc.stdout.readline, ""):
            text = line.rstrip()
            if text:
                print(f"[{name}] {text}", flush=True)
    except Exception as e:
        print(f"[{name}] [输出读取异常] {e}", file=sys.stderr, flush=True)
    finally:
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except Exception:
                pass


def start_process(name: str, cmd: list[str]) -> subprocess.Popen | None:
    """启动子进程并启动输出转发线程。失败返回 None。"""
    # 校验可执行模块是否可导入（避免静默失败）
    try:
        # 不实际启动模块，仅检查 sys.executable 是否存在
        if not Path(sys.executable).exists():
            raise FileNotFoundError(sys.executable)
    except Exception as e:
        print(f"[{name}] [启动失败] 解释器不可用: {e}", file=sys.stderr, flush=True)
        return None

    creationflags = 0
    if sys.platform == "win32":
        # 让子进程独立成组，Ctrl+C 不会直接传给它们，由主进程统一终止
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
            env=_env_with_utf8(),
            creationflags=creationflags,
        )
    except FileNotFoundError as e:
        # 常见于 streamlit / uvicorn 未安装
        print(f"[{name}] [启动失败] 找不到可执行文件: {e}", file=sys.stderr, flush=True)
        print(f"[{name}] [提示] 请先执行: pip install -r requirements.txt", file=sys.stderr, flush=True)
        return None
    except Exception as e:
        print(f"[{name}] [启动失败] {e}", file=sys.stderr, flush=True)
        return None

    t = threading.Thread(target=_stream_output, args=(proc, name), daemon=True)
    t.start()
    PROCESSES.append((name, proc, t))
    print(f"[启动] {name} PID={proc.pid}  命令: {' '.join(cmd)}", flush=True)
    return proc


def _faiss_index_exists() -> bool:
    """判断 FAISS 索引目录中是否存在 index.faiss 文件。"""
    idx_dir = PROJECT_ROOT / "data" / "faiss_index"
    if not idx_dir.is_dir():
        return False
    return (idx_dir / "index.faiss").exists()


def _run_ingest(cmd: list[str], label: str) -> int:
    """同步执行入库命令，打印输出并返回退出码。"""
    print(f"[入库] {label} ...", flush=True)
    try:
        return subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=_env_with_utf8(),
        ).returncode
    except Exception as e:
        print(f"[入库] [异常] {e}", file=sys.stderr, flush=True)
        return -1


def maybe_ingest(force: bool = False) -> None:
    """首次入库：FAISS 索引不存在时，按优先级尝试三个数据源。

    优先级：data/raw 目录 > data/seed/shu_ju.jsonl > 跳过
    """
    if not force and _faiss_index_exists():
        print("[入库] FAISS 索引已存在，跳过首次入库", flush=True)
        return

    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_files = (
        [f for f in raw_dir.rglob("*") if f.is_file() and f.suffix.lower() in {".txt", ".docx", ".pdf"}]
        if raw_dir.is_dir()
        else []
    )

    seed_jsonl = PROJECT_ROOT / "data" / "seed" / "shu_ju.jsonl"

    # 优先级 1：data/raw 里的原始规范文件
    if raw_files:
        rc = _run_ingest(
            [sys.executable, "-m", "scripts.ingest", "data/raw"],
            f"检测到 {len(raw_files)} 个原始规范文件，开始入库",
        )
        if rc != 0:
            print("[入库] [警告] 原始规范入库失败，仍继续启动服务", file=sys.stderr, flush=True)
        else:
            print("[入库] 完成（原始规范）", flush=True)
        return

    # 优先级 2：data/seed/shu_ju.jsonl 种子数据
    if seed_jsonl.is_file():
        rc = _run_ingest(
            [sys.executable, "-m", "scripts.import_jsonl", "data/seed/shu_ju.jsonl"],
            f"data/raw 为空，使用种子数据 {seed_jsonl.name} 入库",
        )
        if rc != 0:
            print("[入库] [警告] 种子数据入库失败，仍继续启动服务", file=sys.stderr, flush=True)
        else:
            print("[入库] 完成（种子数据）", flush=True)
        return

    print("[入库] 未找到任何数据源（data/raw 空 + data/seed/shu_ju.jsonl 不存在），跳过入库", flush=True)


def stop_all() -> None:
    """优雅停止全部子进程：先发 CTRL_BREAK/TERM，5s 后强杀。"""
    if not PROCESSES:
        return
    print("\n[停止] 正在停止全部子进程 ...", flush=True)
    # 倒序停止（先 frontend，再 backend，最后 watcher）
    for name, proc, _ in reversed(PROCESSES):
        if proc.poll() is None:
            try:
                if sys.platform == "win32":
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    proc.terminate()
            except Exception as e:
                print(f"[停止] [警告] 终止 {name} 失败: {e}", file=sys.stderr, flush=True)

    deadline = time.time() + 5.0
    for name, proc, _ in PROCESSES:
        remaining = max(0.1, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            print(f"[停止] [强制] 杀死 {name} PID={proc.pid}", file=sys.stderr, flush=True)
            try:
                proc.kill()
            except Exception:
                pass


def wait_for_exit() -> int:
    """阻塞等待任一子进程退出；返回首个退出的退出码。"""
    while True:
        for name, proc, _ in PROCESSES:
            rc = proc.poll()
            if rc is not None:
                print(f"[退出] {name} 进程结束，退出码={rc}", flush=True)
                return rc
        time.sleep(1.0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ArchiRAG 一键启动（backend + watcher + frontend + 首次入库）",
    )
    parser.add_argument("--skip-ingest", action="store_true", help="跳过首次入库检查")
    parser.add_argument("--skip-watcher", action="store_true", help="不启动文件监控 watcher")
    parser.add_argument("--skip-frontend", action="store_true", help="不启动 Streamlit 前端")
    parser.add_argument("--skip-backend", action="store_true", help="不启动 uvicorn 后端")
    parser.add_argument("--backend-port", type=int, default=8000, help="后端端口（默认 8000）")
    parser.add_argument("--frontend-port", type=int, default=8501, help="前端端口（默认 8501）")
    args = parser.parse_args()

    os.chdir(str(PROJECT_ROOT))

    # 1. 首次入库（同步阻塞，完成后再起服务）
    if not args.skip_ingest:
        maybe_ingest()

    # 2. 启动 watcher
    if not args.skip_watcher:
        start_process("watcher", [sys.executable, "-m", "app.watcher.file_monitor"])

    # 3. 启动后端
    if not args.skip_backend:
        start_process(
            "backend",
            [
                sys.executable, "-m", "uvicorn",
                "app.main:app",
                "--host", "0.0.0.0",
                "--port", str(args.backend_port),
                "--workers", "1",
            ],
        )

    # 4. 启动前端
    if not args.skip_frontend:
        start_process(
            "frontend",
            [
                sys.executable, "-m", "streamlit", "run",
                "frontend/streamlit_app.py",
                f"--server.port={args.frontend_port}",
                "--server.address=0.0.0.0",
                "--server.headless=true",
            ],
        )

    if not PROCESSES:
        print("[错误] 没有任何子进程被启动，退出", file=sys.stderr)
        return 1

    print("\n[就绪] 服务启动中：", flush=True)
    if not args.skip_backend:
        print(f"  后端 API 健康检查: http://localhost:{args.backend_port}/api/health", flush=True)
    if not args.skip_frontend:
        print(f"  前端 UI:           http://localhost:{args.frontend_port}", flush=True)
    print("  按 Ctrl+C 停止全部服务\n", flush=True)

    # 5. 等待子进程退出或 Ctrl+C
    try:
        return wait_for_exit()
    except KeyboardInterrupt:
        stop_all()
        return 0
    finally:
        # 兜底：异常退出时也尝试清理
        stop_all()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        stop_all()
        raise SystemExit(0)
