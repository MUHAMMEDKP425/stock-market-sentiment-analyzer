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
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = finbert_model(**inputs)
    scores = softmax(outputs.logits.detach().numpy()[0])
    labels = ['negative', 'neutral', 'positive']
    
    # Store all scores
    results = {labels[i]: float(scores[i]) for i in range(len(labels))}
    
    # Pick the label with highest score
    sentiment = max(results, key=results.get)
    confidence = results[sentiment] * 100  # convert to percentage
    
    # Display all scores in Streamlit
    st.write(f"**Confidence scores:** {results}")
    
    return sentiment, confidence

# =============================
# 🌐 STREAMLIT APP UI
# =============================
st.title("📈 Stock Market Sentiment Analyzer")
st.markdown("Analyze the sentiment of financial news, tweets, or stock updates using both Logistic Regression and FinBERT AI.")

st.subheader("💬 Enter a financial sentence:")
user_input = st.text_area("Type a financial sentence below:", "")

# --- Logistic Regression Prediction ---
if st.button("Analyze Sentiment (Trained Model)"):
    if user_input.strip():
        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)[0]
        color = "green" if prediction == "positive" else "red" if prediction == "negative" else "gray"
        st.markdown(f"<h3 style='color:{color}'>Predicted Sentiment (Trained): {prediction.upper()} 🎯</h3>", unsafe_allow_html=True)
    else:
        st.warning("Please type something before analyzing.")

# --- FinBERT AI Prediction ---
if st.button("Analyze Sentiment (AI FinBERT) 🤖"):
    if user_input.strip():
        sentiment, confidence = finbert_sentiment(user_input)
        color = "green" if sentiment == "positive" else "red" if sentiment == "negative" else "gray"
        st.markdown(f"<h3 style='color:{color}'>AI Prediction (FinBERT): {sentiment.upper()} 🤖 ({confidence:.1f}% sure)</h3>", unsafe_allow_html=True)
    else:
        st.warning("Please type something before analyzing.")


# --- Chart (optional) ---
st.subheader("📊 Example Sentiment Distribution")
sample_data = pd.DataFrame({
    'label': ['positive', 'negative', 'neutral'],
    'count': [45, 35, 20]
})
fig, ax = plt.subplots()
ax.bar(sample_data['label'], sample_data['count'], color=['green', 'red', 'gray'])
ax.set_title("Example Financial Sentiment Distribution")
st.pyplot(fig)

# --- Footer ---
st.markdown("---")
st.markdown("👨‍💻 Built by **Muhammed KP** — Powered by Streamlit, Logistic Regression & FinBERT 🚀")
