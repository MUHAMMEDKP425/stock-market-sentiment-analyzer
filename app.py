# =============================
# 📈 STOCK MARKET SENTIMENT ANALYZER (with Gemini AI)
# =============================

import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import google.generativeai as genai

# --- SETUP ---
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# --- CONFIGURE GEMINI ---
genai.configure(api_key="AIzaSyAhWPcSXF4R9WfkDT6nbVd_RH-JxbfvV2I")  # 👈 Replace with your actual key

# --- CLEANING FUNCTION ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# --- GEMINI AI SENTIMENT FUNCTION ---
def ai_sentiment_analysis(text):
    prompt = (
        f"Analyze the sentiment of this financial statement. "
        f"Reply only with 'positive', 'negative', or 'neutral': {text}"
    )
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    response = model.generate_content(prompt)
    return response.text.strip().lower()


# =============================
# 🌐 STREAMLIT APP
# =============================
st.title("📈 Stock Market Sentiment Analyzer")
st.markdown("Analyze the sentiment of financial news, tweets, or stock updates.")

st.subheader("💬 Enter text for sentiment prediction:")
user_input = st.text_area("Type a financial sentence...", "")

# --- Dummy training example (to keep app running) ---
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

# --- AI (GEMINI) SENTIMENT BUTTON ---
if st.button("Analyze Sentiment (AI) 🤖"):
    if user_input.strip():
        sentiment = ai_sentiment_analysis(user_input)
        color = "green" if "positive" in sentiment else "red" if "negative" in sentiment else "gray"
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
st.markdown("👨‍💻 Built by **Muhammed KP** — Powered by Streamlit & Gemini AI 🚀")
