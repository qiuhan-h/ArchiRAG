FROM nginx:alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf

# SSL 证书挂载点（compose 中 bind ssl/ 目录）
VOLUME ["/etc/nginx/ssl"]
