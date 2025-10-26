"""
Файл: predict.py
Назначение: использовать обученную LSTM-модель для предсказания токсичности комментариев.

Использование:
    python predict.py "Your comment text here"

Перед запуском убедись, что:
- Файл модели 'lstm_model.pt' находится в той же папке.
- Файл 'practic.py' уже использовался для обучения модели.

Выводит вероятности по категориям:
[toxic, severe_toxic, obscene, threat, insult, identity_hate]
"""

import sys
import torch
from torch import nn

# Импортируем архитектуру из practic.py (если код в одном проекте)
from practic import LSTMSimple, Vocab

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Загружаем модель и словарь
print("\nЗагрузка обученной модели...")
checkpoint = torch.load('lstm_model.pt', map_location='cpu')
model_state = checkpoint['model_state']
vocab_dict = checkpoint['vocab']

# Восстанавливаем модель (размер словаря должен совпадать)
model = LSTMSimple(vocab_size=len(vocab_dict)).to('cpu')
model.load_state_dict(model_state)
model.eval()

# Вспомогательная функция токенизации и кодирования текста
def encode_text(text, vocab, max_len=100):
    tokens = text.lower().split()
    ids = [vocab.get(w, 1) for w in tokens][:max_len]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return torch.tensor([ids], dtype=torch.long)


def predict_toxicity(text):
    x = encode_text(text, vocab_dict)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).squeeze().numpy()
    return probs


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("\nИспользование: python predict.py 'Your comment here'\n")
        sys.exit(0)

    text = sys.argv[1]
    probs = predict_toxicity(text)

    print(f"\nТекст: {text}\n")
    print("Вероятности по категориям:")
    for label, p in zip(LABELS, probs):
        print(f"  {label:15s}: {p:.3f}")

    toxic_score = probs.mean()
    print(f"\nСредний уровень токсичности: {toxic_score:.3f}")

    if toxic_score > 0.5:
        print("⚠️  Комментарий, вероятно, токсичный.")
    else:
        print("✅  Комментарий выглядит нейтральным.")
