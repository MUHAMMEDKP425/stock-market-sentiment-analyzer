# =============================
# 📈 STOCK MARKET SENTIMENT ANALYZER (Trained + FinBERT)
# =============================

import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

# --- SETUP ---
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# --- LOAD TRAINED MODEL AND VECTORIZER ---
st.write("Loading trained sentiment model... ⏳")
tfidf = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("sentiment_model.pkl")
st.write("✅ Logistic Regression model loaded successfully!")

# --- LOAD FINBERT MODEL ---
@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, finbert_model

tokenizer, finbert_model = load_finbert()

# --- CLEANING FUNCTION ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# --- FINBERT ANALYSIS FUNCTION ---
def finbert_sentiment(text):
    plt.clf()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = finbert_model(**inputs)
    scores = softmax(outputs.logits.detach().numpy()[0])

    # ✅ Corrected label order for ProsusAI/finbert
    labels = ['positive', 'negative', 'neutral']

    results = {labels[i]: float(scores[i]) for i in range(len(labels))}
    sentiment = max(results, key=results.get)
    confidence = results[sentiment] * 100

    # --- Keyword correction ---
    text_lower = text.lower()
    positive_keywords = [
        "profit", "profits", "gain", "rise", "growth", "up", "increase", "jump",
        "record profit", "record profits", "strong", "beat", "surged", "higher",
        "improved", "good", "positive", "bullish"
    ]
    negative_keywords = [
        "loss", "drop", "fall", "decline", "down", "weak", "miss",
        "cut", "slump", "negative", "bearish", "bad", "decrease"
    ]

    # Optional correction if model confidence is low
    if sentiment == "neutral" or confidence < 60:
        if any(word in text_lower for word in positive_keywords):
            sentiment, confidence = "positive", 90
        elif any(word in text_lower for word in negative_keywords):
            sentiment, confidence = "negative", 90

    if any(phrase in text_lower for phrase in ["record profit", "record profits", "strong results", "beat expectations"]):
        sentiment, confidence = "positive", 95

    # --- Display FinBERT confidence scores ---
    st.markdown("### 🔍 Confidence Scores")
    st.json(results)

    # --- Plot FinBERT Confidence Chart (Corrected Order) ---
    fig, ax = plt.subplots()
    colors = ["green", "red", "gray"]  # positive, negative, neutral
    ax.bar(results.keys(), results.values(), color=colors)
    ax.set_title("FinBERT Confidence Scores", fontsize=12)
    ax.set_ylabel("Probability", fontsize=10)

    # Highlight predicted sentiment bar
    for i, label in enumerate(results.keys()):
        if label == sentiment:
            ax.patches[i].set_edgecolor("black")
            ax.patches[i].set_linewidth(3)

    st.pyplot(fig, clear_figure=True)

    return sentiment, confidence


# =============================
# 🌐 STREAMLIT APP UI (NO EXTRA GRAPH)
# =============================

st.title("📈 Stock Market Sentiment Analyzer")
st.markdown("Analyze the sentiment of **financial news, tweets, or stock updates** using both **Trained Model (ML)** and **FinBERT AI**.")

st.subheader("💬 Enter a financial sentence:")
user_input = st.text_area("Type a financial sentence below:", "")

# --- Layout: Two Columns ---
col1, col2 = st.columns(2)

# --- LEFT: Logistic Regression Model ---
with col1:
    st.markdown("### 🧠 Trained Model")
    if st.button("Analyze (Trained Model)", key="trained_model"):
        if user_input.strip():
            cleaned = clean_text(user_input)
            vector = tfidf.transform([cleaned])
            prediction = model.predict(vector)[0]

            color = "green" if prediction == "positive" else "red" if prediction == "negative" else "gray"
            bg_color = "#d1ffd6" if prediction == "positive" else "#ffd6d6" if prediction == "negative" else "#f0f0f0"

            st.markdown(
                f"""
                <div style="
                    background-color:{bg_color};
                    padding:15px;
                    border-radius:10px;
                    text-align:center;
                    box-shadow:0 0 8px rgba(0,0,0,0.2);">
                    <h4 style="color:{color};">Predicted Sentiment: {prediction.upper()} 🎯</h4>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("Please type something before analyzing.")

# --- RIGHT: FinBERT AI ---
with col2:
    st.markdown("### 🤖 FinBERT AI")
    if st.button("Analyze (AI FinBERT)", key="ai_model"):
        if user_input.strip():
            sentiment, confidence = finbert_sentiment(user_input)

            color = "green" if sentiment == "positive" else "red" if sentiment == "negative" else "gray"
            bg_color = "#d1ffd6" if sentiment == "positive" else "#ffd6d6" if sentiment == "negative" else "#f0f0f0"

            st.markdown(
                f"""
                <div style="
                    background-color:{bg_color};
                    padding:15px;
                    border-radius:10px;
                    text-align:center;
                    box-shadow:0 0 8px rgba(0,0,0,0.2);">
                    <h4 style="color:{color};">AI Prediction: {sentiment.upper()} 🤖</h4>
                    <p><b>Confidence:</b> {confidence:.1f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("Please type something before analyzing.")

# --- Footer ---
st.markdown("---")
st.markdown("👨‍💻 Built by **Muhammed KP** — Powered by Streamlit, Logistic Regression & FinBERT 🚀")
