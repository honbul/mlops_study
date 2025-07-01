FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

WORKDIR /workspace

# Clone timm repo (이미 준비된 경우 COPY만)
COPY ./code/pytorch-image-models /workspace/timm
RUN pip install -e ./timm[dev]

# 기타 라이브러리 설치 (선택)
#COPY requirements.txt .
#RUN pip install --upgrade pip && pip install -r requirements.txt

# tiny-imagenet 데이터 복사
#COPY ./data/tiny-imagenet-200 /workspace/tiny-imagenet-200

# 작업 기본 디렉토리
WORKDIR /workspace
