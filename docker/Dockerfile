# ArchiRAG 后端
# 默认镜像：轻量（无 torch），Embedding 走确定性降级向量
# 构建真实语义镜像（含 bge-large-zh-v1.5，首次运行需联网下载约 1.3GB 权重）：
#   docker build -f docker/Dockerfile --build-arg INSTALL_EMBED=true -t archirag:bge .
FROM python:3.11-slim

WORKDIR /app

# 系统依赖（faiss/pdf 解析需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 先装核心依赖（利用 Docker 层缓存；国内镜像加速）
COPY requirements.txt requirements-embed.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# 可选：真实 bge 语义向量（torch + sentence-transformers）
# 阿里云镜像加速（国内构建首选；生产服务器在海外可改 download.pytorch.org/whl/cpu）
ARG INSTALL_EMBED=false
RUN if [ "$INSTALL_EMBED" = "true" ]; then \
        pip install --no-cache-dir torch \
            -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com && \
        pip install --no-cache-dir -r requirements-embed.txt \
            -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com ; \
    fi

# 应用代码（WORKDIR=/app 即在 sys.path，无需 pip install -e .）
COPY . .

# 数据卷（规范文件 / FAISS 索引 / SQLite；compose 中 bind 到宿主机 ../data）
VOLUME ["/app/data"]

EXPOSE 8000

# 默认启动 FastAPI（单 worker 避免 watcher 冲突）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
