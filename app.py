# =============================
# 📈 STOCK MARKET SENTIMENT ANALYZER (Trained on FinancialPhraseBank)
# =============================

import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import joblib

# --- SETUP ---
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# --- LOAD TRAINED MODEL AND VECTORIZER ---
st.write("Loading trained sentiment model... ⏳")
tfidf = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("sentiment_model.pkl")
st.write("✅ Model loaded successfully!")

# --- CLEANING FUNCTION ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# =============================
# 🌐 STREAMLIT APP UI
# =============================
st.title("📈 Stock Market Sentiment Analyzer")
st.markdown("Analyze the sentiment of financial news, tweets, or stock updates using a model trained on real financial data.")

st.subheader("💬 Enter a financial sentence:")
user_input = st.text_area("Type a financial sentence below:", "")

# --- NORMAL (TRAINED MODEL) SENTIMENT BUTTON ---
if st.button("Analyze Sentiment"):
    if user_input.strip():
        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)[0]
        color = "green" if prediction == "positive" else "red" if prediction == "negative" else "gray"
        st.markdown(f"<h3 style='color:{color}'>Predicted Sentiment: {prediction.upper()} 🎯</h3>", unsafe_allow_html=True)
    else:
        st.warning("Please type something before analyzing.")

# --- OPTIONAL CHART ---
st.subheader("📊 Example Sentiment Distribution")
sample_data = pd.DataFrame({
    'label': ['positive', 'negative', 'neutral'],
    'count': [45, 35, 20]
})
fig, ax = plt.subplots()
ax.bar(sample_data['label'], sample_data['count'], color=['green', 'red', 'gray'])
ax.set_title("Example Financial Sentiment Distribution")
st.pyplot(fig)

# --- FOOTER ---
st.markdown("---")
st.markdown("👨‍💻 Built by **Muhammed KP** — Powered by Streamlit & FinancialPhraseBank Dataset 🚀")
