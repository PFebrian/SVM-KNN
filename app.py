import streamlit as st
import re
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

# ===============================
# LOAD MODEL & SCALER
# ===============================
svm_model = joblib.load("svm_final.pkl")
knn_model = joblib.load("knn_final.pkl")
scaler = joblib.load("scaler.pkl")

labse = SentenceTransformer("sentence-transformers/LaBSE")

# ===============================
# PREPROCESSING
# ===============================
daftar_abbrev = {
    "jgn": "jangan", "bgt": "banget", "ak": "aku", "ga": "tidak",
    "gk": "tidak", "nggak": "tidak", "gue": "aku", "gw": "aku",
    "lu": "kamu", "lo": "kamu", "anj": "anjing"
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
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)
    return text

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="Deteksi Pelecehan Seksual", layout="centered")

st.title("🔍 Deteksi Komentar Pelecehan Seksual")
st.write("Model: **LaBSE + SVM / KNN**")

kalimat = st.text_area("Masukkan komentar:")

model_choice = st.radio(
    "Pilih algoritma:",
    ["SVM", "KNN"]
)

if st.button("Prediksi"):
    if kalimat.strip() == "":
        st.warning("Komentar tidak boleh kosong.")
    else:
        clean_text = preprocess_text(kalimat)
        embedding = labse.encode([clean_text])
        embedding_std = scaler.transform(embedding)

        if model_choice == "SVM":
            pred = svm_model.predict(embedding_std)[0]
        else:
            pred = knn_model.predict(embedding_std)[0]

        hasil = "🚨 Pelecehan Seksual" if pred == 1 else "✅ Non-Pelecehan"

        st.subheader("Hasil Prediksi:")
        st.success(hasil)