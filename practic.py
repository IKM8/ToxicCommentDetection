"""
Практическая часть проекта: Обнаружение токсичных комментариев (Jigsaw)
Файл: Практическая_часть_—_Обучение_нейросети_на_Jigsaw.py
Авторы: Глушков Н.В, Гарифьянов А.Д, Тарасов Н.Д

Описание:
- Скрипт содержит два режима: 1) LSTM-прототип (быстрый) 2) BERT (HuggingFace, более точный)
- Поддерживает мульти-лейбл классификацию по 6 меткам (toxic, severe_toxic, obscene, threat, insult, identity_hate)
- Сохранение модели и метрик

Инструкция по использованию:
1) Установите зависимости:
   pip install -r requirements.txt
   (в requirements.txt должны быть: torch, transformers, datasets, scikit-learn, pandas, numpy, tqdm, sentencepiece если нужен)

2) Скачайте датасет Jigsaw (например с Kaggle) и поместите csv в папку data/ (имя файла: train.csv)
   Ссылка (пример): https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

3) Запустите скрипт:
   python Практическая_часть_—_Обучение_нейросети_на_Jigsaw.py --mode lstm
   python Практическая_часть_—_Обучение_нейросети_на_Jigsaw.py --mode bert

Примечания:
- Режим 'lstm' подходит для быстрой проверки; 'bert' даёт лучшее качество, но требует GPU и больше времени.
- Скрипт рассчитан на PyTorch backend.

"""

import os
import argparse
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# -------------------- Утилиты --------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------- Загрузка и подготовка данных --------------------

class JigsawDataset(Dataset):
    def __init__(self, texts, labels, tokenizer=None, max_len=128, mode='lstm'):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        if self.mode == 'bert':
            enc = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item['labels'] = torch.tensor(label, dtype=torch.float)
            return item
        else:
            # для LSTM — простая токенизация по пробелу и индексация слов
            tokens = text.lower().split()[:self.max_len]
            return {
                'tokens': tokens,
                'labels': torch.tensor(label, dtype=torch.float)
            }


def load_jigsaw(path='data/train.csv', sample=None):
    df = pd.read_csv(path)
    # Ожидаем, что в датасете есть столбцы: 'comment_text' и 6 меток
    label_cols = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
    df[label_cols] = df[label_cols].fillna(0).astype(int)
    df['comment_text'] = df['comment_text'].fillna('')
    if sample is not None:
        df = df.sample(sample, random_state=42).reset_index(drop=True)
    return df


# -------------------- Простой LSTM-прототип --------------------

class Vocab:
    def __init__(self, min_freq=2):
        self.token2idx = {'<pad>':0,'<unk>':1}
        self.idx2token = ['<pad>','<unk>']
        self.min_freq = min_freq
        self.freq = {}

    def build(self, texts):
        for t in texts:
            for w in t.lower().split():
                self.freq[w] = self.freq.get(w,0)+1
        for w, f in self.freq.items():
            if f >= self.min_freq:
                self.token2idx[w] = len(self.idx2token)
                self.idx2token.append(w)

    def encode(self, tokens, max_len):
        ids = [self.token2idx.get(w,1) for w in tokens]
        if len(ids) < max_len:
            ids = ids + [0]*(max_len-len(ids))
        else:
            ids = ids[:max_len]
        return ids


class LSTMSimple(nn.Module):
    def __init__(self, vocab_size, emb_dim=100, hidden=128, num_labels=6, pad_idx=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hidden*2, num_labels)

    def forward(self, x):
        # x: (batch, seq_len)
        e = self.emb(x)
        out, _ = self.lstm(e)
        out = out.permute(0,2,1)  # (batch, hidden*2, seq_len)
        pooled = self.pool(out).squeeze(-1)
        logits = self.fc(pooled)
        return logits


def train_lstm(df, device, epochs=3, batch_size=64, max_len=100, sample=None):
    # Подготовка
    texts = df['comment_text'].tolist()
    labels = df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].values
    vocab = Vocab(min_freq=2)
    vocab.build(texts[:100000] if len(texts)>100000 else texts)

    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.1, random_state=42)
    train_ds = JigsawDataset(X_train, y_train, tokenizer=None, max_len=max_len, mode='lstm')
    val_ds = JigsawDataset(X_val, y_val, tokenizer=None, max_len=max_len, mode='lstm')

    def collate(batch):
        seqs = [vocab.encode(b['tokens'], max_len) for b in batch]
        seqs = torch.tensor(seqs, dtype=torch.long)
        labs = torch.stack([b['labels'] for b in batch])
        return seqs, labs

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model = LSTMSimple(len(vocab.idx2token)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs+1):
        model.train()
        pbar = tqdm(train_dl, desc=f"Train epoch {epoch}")
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix({'loss': loss.item()})

        # валидация
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.append(probs)
                trues.append(yb.cpu().numpy())
        preds = np.vstack(preds)
        trues = np.vstack(trues)
        pred_bin = (preds >= 0.5).astype(int)
        f1 = f1_score(trues, pred_bin, average='micro')
        print(f"Epoch {epoch} — val F1 (micro): {f1:.4f}")

    torch.save({'model_state': model.state_dict(), 'vocab': vocab.token2idx}, 'lstm_model.pt')
    print('LSTM модель сохранена как lstm_model.pt')


# -------------------- BERT (HuggingFace) --------------------

def train_bert(df, device, model_name='bert-base-uncased', epochs=2, batch_size=16, max_len=128):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

    label_cols = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
    df_train, df_val = train_test_split(df, test_size=0.1, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_texts = df_train['comment_text'].tolist()
    train_labels = df_train[label_cols].values.tolist()
    val_texts = df_val['comment_text'].tolist()
    val_labels = df_val[label_cols].values.tolist()

    class HF_Dataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
            return item

    train_dataset = HF_Dataset(train_texts, train_labels, tokenizer, max_len)
    val_dataset = HF_Dataset(val_texts, val_labels, tokenizer, max_len)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6, problem_type='multi_label_classification')

    # TrainingArguments работают с метриками через compute_metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1/(1+np.exp(-logits))
        preds = (probs >= 0.5).astype(int)
        f1 = f1_score(labels, preds, average='micro')
        return {'f1_micro': f1}

    args = TrainingArguments(
        output_dir='bert_output',
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model('bert_model')
    print('BERT модель сохранена в папке bert_model')


# -------------------- Основная точка входа --------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['lstm','bert'], default='lstm')
    parser.add_argument('--data', default='data/train.csv')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_len', type=int, default=128)
    args = parser.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Используется устройство: {device}')

    df = load_jigsaw(args.data)
    # Для ускорения экспериментов можно взять подвыборку: df = df.sample(20000, random_state=42)

    if args.mode == 'lstm':
        train_lstm(df, device, epochs=args.epochs, batch_size=args.batch_size, max_len=args.max_len)
    else:
        train_bert(df, device, model_name='bert-base-uncased', epochs=args.epochs, batch_size=args.batch_size, max_len=args.max_len)
