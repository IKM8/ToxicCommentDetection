import streamlit as st
import torch
import numpy as np
import re

st.set_page_config(page_title="Обнаружение токсичных комментариев", layout="wide")

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
DEFAULT_THRESHOLDS = {
    'toxic': 0.5,
    'severe_toxic': 0.4,
    'obscene': 0.6,
    'threat': 0.6,
    'insult': 0.5,
    'identity_hate': 0.5
}
PROFANITY = ['idiot', 'fuck', 'stupid', 'bastard', 'shit', 'ass']

# ---------------------- загрузка модели ----------------------
@st.cache_resource
def load_model(path='lstm_model.pt'):
    checkpoint = torch.load(path, map_location='cpu')
    model_state = checkpoint['model_state']
    vocab = checkpoint['vocab']
    from torch import nn
    class LSTMSimple(nn.Module):
        def __init__(self, vocab_size, emb_dim=100, hidden=128, num_labels=6, pad_idx=0):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
            self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
            self.pool = nn.AdaptiveMaxPool1d(1)
            self.fc = nn.Linear(hidden*2, num_labels)
        def forward(self, x):
            e = self.emb(x)
            out, _ = self.lstm(e)
            out = out.permute(0,2,1)
            pooled = self.pool(out).squeeze(-1)
            logits = self.fc(pooled)
            return logits

    model = LSTMSimple(vocab_size=len(vocab))
    model.load_state_dict(model_state)
    model.eval()
    return model, vocab

model, VOCAB = load_model()

# ---------------------- вспомогательные функции ----------------------
def encode_text_single(text, vocab, max_len=100):
    tokens = text.lower().split()
    ids = [vocab.get(w, 1) for w in tokens][:max_len]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return torch.tensor([ids], dtype=torch.long)

def predict_probs(texts):
    xs = [encode_text_single(t, VOCAB) for t in texts]
    x = torch.cat(xs, dim=0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).numpy()
    return probs

def highlight_text(text):
    html = text
    for w in set(PROFANITY):
        pattern = re.compile(re.escape(w), re.IGNORECASE)
        html = pattern.sub(f"<mark>{w}</mark>", html)
    return html

# ---------------------- интерфейс ----------------------
st.title("🧠 Обнаружение токсичных комментариев")
st.markdown("Введите комментарий ниже, чтобы определить, какие типы токсичности в нём присутствуют.")
st.markdown("---")
st.markdown("**Authors:** Glushkov Nikita, Tarasov Nikita, Garifyanov Amir")
st.markdown("**Contact:** glushkow.nikita7v@gmail.com")
st.markdown("---")

text = st.text_area("Введите комментарий:", height=120)
st.subheader("Пороги меток")
thresh_cols = st.columns(3)
thresholds = {}
for i, lab in enumerate(LABELS):
    with thresh_cols[i % 3]:
        thresholds[lab] = st.slider(lab, 0.0, 1.0, float(DEFAULT_THRESHOLDS[lab]), 0.01)

if st.button("Анализировать"):
    if not text.strip():
        st.warning("Введите текст комментария!")
    else:
        probs = predict_probs([text]).squeeze()
        st.subheader("Результаты классификации:")
        cols = st.columns(3)
        for i, label in enumerate(LABELS):
            with cols[i % 3]:
                if probs[i] >= thresholds[label]:
                    st.success(f"✅ {label} ({probs[i]:.2f})")
                else:
                    st.error(f"❌ {label} ({probs[i]:.2f})")

        # Уровень токсичности по максимуму ключевых меток
        score = float(np.max(probs[[0,1,2,3,4]]))
        st.progress(min(max(score, 0.0), 1.0))
        if score >= 0.5:
            st.error(f"⚠️ Комментарий, вероятно, токсичный (уровень: {score:.2f})")
        else:
            st.success(f"✅ Комментарий выглядит нейтральным (уровень: {score:.2f})")