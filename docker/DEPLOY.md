## ArchiRAG 生产部署指南

### 前置条件

1. Docker + Docker Compose 已安装
2. 域名已解析到服务器 IP
3. SSL 证书（`.pem` 格式，含 fullchain + privkey）
4. 阿里云 DashScope API Key（[申请地址](https://dashscope.console.aliyun.com/)）

---

### 步骤 1：配置环境变量

```bash
cd ArchiRAG
cp .env.prod.example .env
```

编辑 `.env`，填写：

```ini
QWEN_API_KEY=sk-xxxxx          # DashScope API Key
MYSQL_ROOT_PASSWORD=强密码       # MySQL root 密码
REDIS_PASSWORD=强密码            # Redis 密码（留空则无密码）
```

同时修改 `MYSQL_URL` 和 `REDIS_URL` 中的 `CHANGE_ME` 为上面设的密码：

```ini
MYSQL_URL=mysql+pymysql://root:你的密码@db:3306/building_rag
REDIS_URL=redis://:你的密码@redis:6379/0
```

### 步骤 2：放置 SSL 证书

```bash
mkdir -p docker/nginx/ssl
cp /path/to/fullchain.pem docker/nginx/ssl/
cp /path/to/privkey.pem docker/nginx/ssl/
```

### 步骤 3：修改 Nginx 域名

编辑 [docker/nginx/nginx.conf](file:///docker/nginx/nginx.conf)，将 `server_name _;` 改为你的真实域名。

### 步骤 4：构建镜像（含 bge 真实语义模型）

```bash
cd docker
INSTALL_EMBED=true docker compose build
```

> 首次构建约 10-15 分钟（下载 torch CPU + bge 权重 ~2GB）。

### 步骤 5：启动服务

```bash
# 生产模式（含 Nginx + HTTPS）
docker compose --profile prod up -d

# 或本地调试（无 Nginx，直连端口）
docker compose up -d
```

### 步骤 6：导入数据（bge 语义向量）

```bash
# 清空旧索引（如从降级模式升级）
rm -rf data/faiss_index/*

# 容器内导入
docker compose run --rm backend python -m scripts.import_jsonl data/seed/shu_ju.jsonl
```

### 步骤 7：验证

```bash
# 健康检查
curl https://你的域名/api/health

# 期望返回：db_type=mysql, cache_type=redis, llm_mode=qwen, embedder=bge
```

---

### 常用运维命令

```bash
# 查看日志
docker compose logs -f backend

# 重启某个服务
docker compose restart backend

# 更新数据
docker compose run --rm backend python -m scripts.import_jsonl data/seed/shu_ju.jsonl

# 停止全部
docker compose down

# 停止并清除数据卷（⚠️ 慎用）
docker compose down -v
```

### 端口说明

| 端口 | 服务 | 生产是否暴露 |
|------|------|-------------|
| 443 | Nginx HTTPS | ✅ |
| 80 | Nginx HTTP→301 | ✅ |
| 8000 | FastAPI | ❌（经 Nginx 反代） |
| 8501 | Streamlit | ❌（经 Nginx 反代） |
| 3306 | MySQL | ❌（仅容器内） |
| 6379 | Redis | ❌（仅容器内） |

> 生产部署后建议在防火墙关闭 8000/8501/3306/6379 端口，只开放 80/443。
