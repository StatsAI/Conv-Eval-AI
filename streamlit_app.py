import streamlit as st
import pandas as pd
import plotly.express as px
import json
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# --- Page Configuration ---
st.set_page_config(page_title="High-Speed Identity API", layout="wide")

# --- ultra-fast Model Training (Happens once on load) ---
@st.cache_resource
def load_fast_model():
    # Synthetic training data to define "Identity Signals"
    # In a production environment, you would load a pre-trained pickle file here.
    data = [
        ("I am confident in my skills and lead the team well", "self-confident"),
        ("I am not sure if I can do this, I feel lost", "uncertain"),
        ("I love learning new things and growing every day", "growth-oriented"),
        ("I analyze every detail and look at the data", "analytical"),
        ("I enjoy painting, writing, and creating new worlds", "creative"),
        ("I don't believe the hype, I need proof first", "skeptical"),
        ("I prefer to stay safe and avoid any big changes", "risk-averse"),
        ("I am building a new career and evolving my mindset", "growth-oriented"),
        ("The numbers don't add up, let's re-calculate", "analytical")
    ]
    texts, labels = zip(*data)
    
    # Simple, high-speed ML Pipeline (TF-IDF + Naive Bayes)
    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB())
    ])
    model.fit(texts, labels)
    return model

fast_model = load_fast_model()

# --- App UI ---
st.title("⚡ Ultra-Fast Identity Evaluation")
st.caption("Target Latency: < 200ms (Current Model: Naive Bayes)")

default_text = (
    "I've been reflecting on my career lately. I used to be very risk-averse, "
    "but lately, I find myself excited by the challenge of building new things from scratch. "
    "I'm becoming much more of a growth-oriented leader than I was a year ago."
)

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Min Confidence Score", 0.0, 1.0, 0.15)

# --- Input Section ---
user_input = st.text_area("Conversation Input:", value=default_text, height=150)
analyze_button = st.button("Analyze Identity Signals", type="primary")

# --- High-Speed Inference ---
if analyze_button and user_input:
    start_time = time.time()
    
    # Get probability distribution across all known classes
    probs = fast_model.predict_proba([user_input])[0]
    classes = fast_model.classes_
    
    # Calculate Latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Format results
    signals = [
        {"Signal": label, "Confidence": round(prob, 4)}
        for label, prob in zip(classes, probs)
        if prob >= threshold
    ]
    signals = sorted(signals, key=lambda x: x['Confidence'], reverse=True)
    
    # Store in session state
    st.session_state['fast_res'] = {"signals": signals, "latency": latency_ms}

# --- Results Display ---
if 'fast_res' in st.session_state:
    res = st.session_state['fast_res']
    
    st.metric("Inference Latency", f"{res['latency']:.2f} ms", delta="-180ms vs LLM")
    
    if res['signals']:
        df = pd.DataFrame(res['signals'])
        
        # Chart
        fig = px.bar(
            df, x="Confidence", y="Signal", 
            orientation='h', color="Confidence",
            color_continuous_scale="Viridis",
            text_auto='.2f'
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Export
        json_data = json.dumps(res, indent=2)
        st.download_button("📥 Download JSON", data=json_data, file_name="fast_eval.json")
    else:
        st.warning("No signals met the threshold.")
