import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import google.generativeai as genai


# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# Configure Gemini API key
genai.configure(api_key="AIzaSyDP07BuN7Mb77RYVjiTwBQUa_jSRcWmG_0")


# --- Define Text Cleaning Function ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)
def ai_sentiment_analysis(text):
    prompt = f"Analyze the sentiment of this financial statement. Reply only with 'positive', 'negative', or 'neutral': {text}"
    response = genai.GenerativeModel("gemini-pro").generate_content(prompt)
    sentiment = response.text.strip().lower()
    return sentiment


# --- Streamlit UI ---
st.title("📈 Stock Market Sentiment Analyzer")
st.markdown("Analyze the sentiment of financial news, tweets, or stock updates.")

# --- Load Sample Data (if not uploaded) ---
st.sidebar.header("Upload your CSV file (optional)")
uploaded_file = st.sidebar.file_uploader("Upload CSV with a 'tweet' column", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
else:
    # Sample dataset (tiny example for demo)
    data = pd.DataFrame({
        'tweet': [
            'Stock prices are rising steadily.',
            'Market crash is worrying investors.',
            'Investors are waiting for news updates.'
        ],
        'label': ['positive', 'negative', 'neutral']
    })

# --- Clean the data ---
data['clean_text'] = data['tweet'].apply(clean_text)

# --- Convert text to vectors ---
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data['clean_text'])
y = data['label']

# --- Train simple model ---
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# --- Text box for live prediction ---
st.subheader("💬 Enter text for sentiment prediction:")
user_input = st.text_area("Type a financial sentence...", "")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)[0]
        st.success(f"Predicted Sentiment: **{prediction.upper()}** 🎯")
    else:
        st.warning("Please type some text before analyzing.")
 if st.button("Analyze Sentiment (AI) 🤖"):
    if user_input.strip():
        sentiment = ai_sentiment_analysis(user_input)
        color = "green" if "positive" in sentiment else "red" if "negative" in sentiment else "gray"
        st.markdown(f"<h3 style='color:{color}'>AI Prediction: {sentiment.upper()} 🤖</h3>", unsafe_allow_html=True)
    else:
        st.warning("Please type something before analyzing.")


# --- Sentiment Distribution Chart ---
st.subheader("📊 Sentiment Distribution in Dataset")
fig, ax = plt.subplots()
data['label'].value_counts().plot(kind='bar', ax=ax, color=['green','red','gray'])
st.pyplot(fig)
