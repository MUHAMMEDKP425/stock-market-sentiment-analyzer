# =============================
# 📈 STOCK MARKET SENTIMENT ANALYZER (Offline FinBERT Version)
# =============================

import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

# --- SETUP ---
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# --- LOAD FINBERT MODEL (No API Needed) ---
@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

tokenizer, finbert_model = load_finbert()

# --- CLEANING FUNCTION ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# --- FINBERT SENTIMENT FUNCTION ---
def ai_sentiment_analysis(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = finbert_model(**inputs)
    scores = softmax(outputs.logits.detach().numpy()[0])
    labels = ['negative', 'neutral', 'positive']
    sentiment = labels[scores.argmax()]
    return sentiment

# =============================
# 🌐 STREAMLIT APP
# =============================
st.title("📈 Stock Market Sentiment Analyzer")
st.markdown("Analyze the sentiment of financial news, tweets, or stock updates.")

st.subheader("💬 Enter text for sentiment prediction:")
user_input = st.text_area("Type a financial sentence...", "")

# --- Dummy training example (for ML baseline) ---
sample_data = pd.DataFrame({
    'tweet': [
        'Stock prices are rising steadily.',
        'Market crash is worrying investors.',
        'Investors are waiting for news updates.'
    ],
    'label': ['positive', 'negative', 'neutral']
})
sample_data['clean_text'] = sample_data['tweet'].apply(clean_text)

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(sample_data['clean_text'])
y = sample_data['label']

model = LogisticRegression(max_iter=200)
model.fit(X, y)

# --- NORMAL (ML) SENTIMENT BUTTON ---
if st.button("Analyze Sentiment"):
    if user_input.strip():
        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)[0]
        color = "green" if prediction == "positive" else "red" if prediction == "negative" else "gray"
        st.markdown(f"<h3 style='color:{color}'>Predicted Sentiment: {prediction.upper()} 🎯</h3>", unsafe_allow_html=True)
    else:
        st.warning("Please type something before analyzing.")

# --- AI (FINBERT) SENTIMENT BUTTON ---
if st.button("Analyze Sentiment (AI) 🤖"):
    if user_input.strip():
        sentiment = ai_sentiment_analysis(user_input)
        color = "green" if sentiment == "positive" else "red" if sentiment == "negative" else "gray"
        st.markdown(f"<h3 style='color:{color}'>AI Prediction: {sentiment.upper()} 🤖</h3>", unsafe_allow_html=True)
    else:
        st.warning("Please type something before analyzing.")

# --- CHART (optional) ---
st.subheader("📊 Sample Sentiment Distribution")
fig, ax = plt.subplots()
sample_data['label'].value_counts().plot(kind='bar', ax=ax, color=['green', 'red', 'gray'])
st.pyplot(fig)

# --- FOOTER ---
st.markdown("---")
st.markdown("👨‍💻 Built by **Muhammed KP** — Powered by Streamlit & FinBERT 🚀")
