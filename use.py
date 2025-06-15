import torch
from transformers import BertForSequenceClassification, AutoTokenizer

# device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 라벨 매핑
id2label = {
    0: "공포", 1: "놀람", 2: "분노", 3: "슬픔",
    4: "중립", 5: "행복", 6: "혐오"
}

# 모델과 토크나이저 로드
model_path = "./results"  # 학습된 모델 저장 위치
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model.to(device) # type: ignore
model.eval()


# 예측 함수
def predict_sentiment(sentence):
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
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
        return id2label[pred] # type: ignore

# 실시간 입력 루프
if __name__ == "__main__":
    print("문장을 입력하세요. 종료하려면 'exit' 또는 'quit'을 입력하세요.\n")
    while True:
        sentence = input("입력 문장: ").strip()
        if sentence.lower() in ['exit', 'quit']:
            print("종료합니다.")
            break
        if sentence == "":
            continue
        prediction = predict_sentiment(sentence)
        print(f"예측 감정: {prediction}\n")
