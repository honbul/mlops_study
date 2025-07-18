FROM python:3.10-slim

# 1. curl 설치 (필수)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# 2. mc 바이너리 설치
RUN mkdir -p /opt/minio-binaries \
 && curl -L https://dl.min.io/client/mc/release/linux-amd64/mc \
      -o /opt/minio-binaries/mc \
 && chmod +x /opt/minio-binaries/mc \
 && ln -s /opt/minio-binaries/mc /usr/local/bin/mc

ENV PATH="/opt/minio-binaries:${PATH}"

# (필요시 추가 라이브러리 설치)
# RUN pip install --no-cache-dir pandas numpy

CMD ["python3"]
