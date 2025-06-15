import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertForSequenceClassification, BertModel, AutoModel
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW 
import pandas as pd
from datasets import Dataset as HFDataset
from tqdm import tqdm
from torch.multiprocessing import freeze_support
import os  # 상단에 import 추가
# 1. 데이터 전처리 및 로드 함수
def load_and_preprocess_data(file_path):
    df = pd.read_excel(file_path)
    emotion_map = {
        "공포": 0, "놀람": 1, "분노": 2, "슬픔": 3,
        "중립": 4, "행복": 5, "혐오": 6
    }
    df['Emotion'] = df['Emotion'].map(emotion_map)
    # data_dict_list = [{"text": text, "label": str(label)} for text, label in zip(df['Sentence'], df['Emotion'])]
    data_dict_list = [{"text": text, "label": label} for text, label in zip(df['Sentence'], df['Emotion'])]
    dataset = HFDataset.from_list(data_dict_list)
    return dataset.train_test_split(test_size=0.2, seed=32)

# 2. Dataset 클래스 정의
class HF_BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, tokenizer, max_len):
        self.dataset = dataset
        self.sent_idx = sent_idx
        self.label_idx = label_idx
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sentence = self.dataset[idx][self.sent_idx]
        label = int(self.dataset[idx][self.label_idx])
        encoding = self.tokenizer(
            sentence,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True,
            return_token_type_ids=True
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # print("input_ids:", encoding['input_ids'])
        # print("decoded:", tokenizer.decode(encoding['input_ids'][0]))
        
        token_type_ids = torch.zeros_like(input_ids)  # 세그먼트 ID가 필요 없는 경우 0으로 채움
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': torch.tensor(label, dtype=torch.long),
            'sentence': sentence  # ✅ 문장 추가
        }

# 3. 모델 클래스 정의 (필요시)
class HF_BERTClassifier(nn.Module):
    def __init__(self, bert_model_name='skt/kobert-base-v1', hidden_size=768, num_classes=7, dr_rate=None):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(p=dr_rate) if dr_rate else None

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            return_dict=True)
        pooled_output = outputs.pooler_output
        if self.dropout:
            pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# 4. 정확도 계산 함수
def calc_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()

# 5. 학습 함수
def train_epoch(model, dataloader, optimizer, scheduler, device, loss_fn, max_grad_norm, log_interval):
    model.train()
    total_loss, total_acc = 0, 0
    max_label = -1
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        batch_max = labels.max().item()
        max_label = max(max_label, batch_max)
        print(f"최대 라벨 값: {max_label}")

        if batch_idx == 0:
            print(f"\n===== DEBUG INFO (Batch {batch_idx}) =====")
            print(f"input_ids dtype: {input_ids.dtype}")
            print(f"input_ids max: {input_ids.max().item()}")
            # print(f"vocab size: {model.bert.config.vocab_size}")
            print(f"token_type_ids unique: {torch.unique(token_type_ids)}")
            print(f"input_ids[0]: {input_ids[0]}")
            print(f"attention_mask[0]: {attention_mask[0]}")
            print(f"token_type_ids[0]: {token_type_ids[0]}")
            print(f"label: {labels[0]}")
            print("==========================================")

        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs.logits
        loss = loss_fn(logits, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        acc = calc_accuracy(logits, labels)
        total_loss += loss.item()
        total_acc += acc

        if batch_idx % log_interval == 0:
            sample_input_ids = input_ids[0].detach().cpu()
            sample_sentence = tokenizer.decode(sample_input_ids, skip_special_tokens=True)
            print(f"📘 학습 중인 문장 예시 (Batch {batch_idx}): {batch['sentence'][0]}")
            print(f"Batch {batch_idx} - Loss: {total_loss/(batch_idx+1):.4f}, Accuracy: {total_acc/(batch_idx+1):.4f}")
            
    return total_loss / len(dataloader), total_acc / len(dataloader)

# 6. 평가 함수
def evaluate(model, dataloader, device, loss_fn):
    model.eval()
    total_loss, total_acc = 0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits = outputs.logits
            loss = loss_fn(logits, labels)

            
            # batch_max = labels.max().item()
            # max_label = max(max_label, batch_max)
            # print(f"최대 라벨 값: {max_label}")

            acc = calc_accuracy(logits, labels)
            total_loss += loss.item()
            total_acc += acc
    return total_loss / len(dataloader), total_acc / len(dataloader)

# 7. 전체 학습 프로세스 함수
def run_training(model, train_loader, test_loader, optimizer, scheduler, device, loss_fn,
                 num_epochs, max_grad_norm, log_interval, output_dir="./results", patience=3):

    model.to(device)
    best_val_acc = 0
    patience_counter = 0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for epoch in range(num_epochs):
        print(f"\n🔁 Epoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device, loss_fn, max_grad_norm, log_interval)
        print(f"✅ Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        val_loss, val_acc = evaluate(model, test_loader, device, loss_fn)
        print(f"🧪 Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early Stopping 로직
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # 모델 저장
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            print(f"💾 Best model saved with val acc: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            print(f"⏳ Early stopping patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("🛑 Early stopping triggered. Stopping training.")
                break
    
    # 모델 저장
    model.save_pretrained(output_dir)

    # tokenizer 저장 (filename_prefix 인자 없이 직접 호출)
    tokenizer.save_vocabulary(output_dir)
    print("📦 토크나이저 저장 완료")


# 8. main 함수 (실행 진입점)
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 로드 및 분할
    split_dataset = load_and_preprocess_data("./content/한국어_단발성_대화_데이터셋.xlsx")
    dataset_train = split_dataset['train']
    dataset_test = split_dataset['test']

    # 토크나이저와 모델 로드
    global tokenizer
    # tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1", trust_remote_code=True)
    # model = HF_BERTClassifier(bert_model_name='skt/kobert-base-v1', num_classes=7, dr_rate=0.1)
    # model = AutoModel.from_pretrained("monologg/kobert")
    model = BertForSequenceClassification.from_pretrained("monologg/kobert", num_labels=7)
    tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
    output_dir = "./results"
    model_path = os.path.join(output_dir, 'best_model.pt')    

    # 모델 파라미터 로드 (기존 모델이 존재하면)
    # if os.path.exists(model_path):
    #     print("🔄 기존 학습된 모델을 불러옵니다...")
    #     model.load_state_dict(torch.load(model_path, map_location=device))
    # else:
    #     print("🆕 새 모델로 학습을 시작합니다.")

    # 데이터셋 및 로더 구성
    max_len = 64
    batch_size = 64
    data_train = HF_BERTDataset(dataset_train, 'text', 'label', tokenizer, max_len)
    data_test = HF_BERTDataset(dataset_test, 'text', 'label', tokenizer, max_len)
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 옵티마이저 및 스케줄러 설정
    learning_rate = 1e-5
    num_epochs = 1
    warmup_ratio = 0.1
    max_grad_norm = 1
    log_interval = 100

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    
    t_total = len(train_loader) * num_epochs
    warmup_step = int(t_total * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)
    loss_fn = nn.CrossEntropyLoss()

    # 학습 시작
    run_training(model, train_loader, test_loader, optimizer, scheduler, device, loss_fn,
                 num_epochs, max_grad_norm, log_interval, output_dir=output_dir,patience=3)

if __name__ == "__main__":
    freeze_support()
    main()
