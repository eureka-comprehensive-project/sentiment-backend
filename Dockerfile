FROM python:3.11-slim

WORKDIR /sentiment

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .  # ✅ 현재 폴더 전체 복사 (app 대신)

EXPOSE 8088

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8088"]