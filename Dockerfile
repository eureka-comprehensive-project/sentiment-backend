FROM python:3.11-slim

WORKDIR /sentiment

COPY requirements.txt .

# torch 먼저 명시적으로 설치 (CPU 버전)
RUN pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cpu

# 나머지 requirements 설치
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8088

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8088"]