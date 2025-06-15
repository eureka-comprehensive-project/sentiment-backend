FROM python:3.11-slim

WORKDIR /sentiment

COPY requirements.txt .

# 의존성 설치 (torch도 포함돼야 함)
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app

EXPOSE 8088

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8088"]

