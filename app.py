import streamlit as st
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Deteksi Pelecehan Verbal (SVM)",
    layout="centered"
)

# =========================
# Preprocessing
# =========================
daftar_abbrev = {
    "jgn": "jangan","bgt": "banget","bngt": "banget","ak": "aku","kl": "kalo","bkn": "bukan","bs" : "bisa",
    "yg": "yang","tdk": "tidak","gmn": "gimana","emg": "emang","sm": "sama","org": "orang","krn": "karena",
    "dgn": "dengan","dr": "dari","jg": "juga","izn": "izin","udh": "udah","bgt": "banget","jdi": "jadi",
    "ap" : "apa","ga": "tidak","g" : "tidak","gk": "tidak","nggak": "tidak","engga": "tidak","aja": "saja",
    "gue" : "aku", "gw": "aku", "lu": "kamu", "lo": "kamu", "gua" : "aku", "anj":"anjing",
    "idc": "i dont care","gws": "get well soon","rn": "right now","idk": "i dont know","btw": "by the way",
    "omg": "oh my god","lmao": "laughing my ass off","lmfao": "laughing my fucking ass off","smh": "shaking my head",
    "tbh": "to be honest","tbk": "to be kind","thx": "thanks","ty": "thank you","tyvm": "thank you very much",
    "ikr": "i know, right","asap": "as soon as possible","rlly": "really","plz": "please","wtf": "what the fuck"
}

def abbrev(text):
    tokens = text.split()
    tokens = [daftar_abbrev.get(t, t) for t in tokens]
    return " ".join(tokens)

def preprocess_text(text):
    text = text.lower()
    text = abbrev(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\S+|#\S+", "", text)
    text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =========================
# Load Dataset
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("tiktok_comments.csv", sep=";", encoding="utf-8")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(float).astype(int)
    df["komentar_bersih"] = df["komentar"].astype(str).apply(preprocess_text)
    df = df[df["komentar_bersih"].str.strip() != ""]
    return df

# =========================
# Load Model LaBSE
# =========================
@st.cache_resource
def load_labse():
    return SentenceTransformer("sentence-transformers/LaBSE")

# =========================
# UI Header
# =========================
st.markdown("""
<h1 style='text-align:center; font-size:34px;'>
🚨 Deteksi Pelecehan Seksual Verbal
</h1>
<p style='text-align:center;'>
Model: <b>SVM RBF + LaBSE</b>
</p>
""", unsafe_allow_html=True)

# =========================
# Load Everything
# =========================
df = load_data()
labse = load_labse()

@st.cache_data
def encode_corpus(texts):
    return labse.encode(texts, show_progress_bar=False)

X = labse.encode(df["komentar_bersih"].tolist(), show_progress_bar=False)
y = df["label"].values

# =========================
# Split FINAL MODEL
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# Scaling
# =========================
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# =========================
# Train SVM FINAL
# =========================
svm_final = SVC(
    kernel="rbf",
    C=1.5,
    gamma="scale",
    class_weight="balanced",
    random_state=42
)

svm_final.fit(X_train_std, y_train)

# =========================
# Input Form
# =========================
st.markdown("### ✍️ Masukkan Komentar")

with st.form("form_prediksi"):
    komentar = st.text_area(
        "Komentar TikTok",
        placeholder="contoh: sexy banget sih dia 😭"
    )
    submit = st.form_submit_button("🔍 Prediksi")

# =========================
# Prediction
# =========================
if submit and komentar.strip() != "":
    clean = preprocess_text(komentar)
    emb = labse.encode([clean])
    emb_std = scaler.transform(emb)
    pred = svm_final.predict(emb_std)[0]

    label = "🚨 **Pelecehan**" if pred == 1 else "✅ **Non-Pelecehan**"

    st.subheader("Hasil Prediksi")
    if pred == 1:
        st.error(label)
    else:
        st.success(label)

# =========================
# Evaluation
# =========================
with st.expander("📊 Lihat Evaluasi Model"):
    y_pred = svm_final.predict(X_test_std)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.markdown(f"- **Akurasi** : `{acc:.2f}`")
    st.markdown(f"- **Precision** : `{prec:.2f}`")
    st.markdown(f"- **Recall** : `{rec:.2f}`")
    st.markdown(f"- **F1-Score** : `{f1:.2f}`")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Pelecehan", "Pelecehan"],
        yticklabels=["Non-Pelecehan", "Pelecehan"],
        ax=ax
    )
    ax.set_xlabel("Prediksi Model")
    ax.set_ylabel("Label Aktual")
    ax.set_title("Confusion Matrix SVM RBF")

    st.pyplot(fig)

# =========================
# Footer
# =========================
st.markdown("---")
st.caption("© 2025 | Deteksi Pelecehan Seksual Verbal • SVM RBF + LaBSE | Peni")

