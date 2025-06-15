from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertForSequenceClassification, AutoTokenizer

# FastAPI 앱 생성
app = FastAPI()

# 요청 바디 모델 정의
class MessageRequest(BaseModel):
    message: str

# 감정 라벨 매핑
id2label = {
    0: "공포", 1: "놀람", 2: "분노", 3: "슬픔",
    4: "중립", 5: "행복", 6: "혐오"
}

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 및 토크나이저 로드
model_path = "./results"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model.to(device)
model.eval()

# 예측 함수
def predict_sentiment(sentence: str) -> str:
    encoding = tokenizer(
        sentence,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    token_type_ids = torch.zeros_like(input_ids).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
        return id2label[pred]

# 루트 확인용 엔드포인트
@app.get("/")
def read_root():
    return {"message": "감정 분석 API에 오신 것을 환영합니다!"}

# 감정 분석 엔드포인트
@app.post("/sentiment/api/predict")
def predict(request: MessageRequest):
    sentiment = predict_sentiment(request.message)
    return {"sentiment": sentiment}
