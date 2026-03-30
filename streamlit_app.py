import streamlit as st
import pandas as pd
import plotly.express as px
import json
import time
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# --- Page Configuration ---
st.set_page_config(page_title="Identity Evaluation Suite", layout="wide")

# --- Model 1: High-Speed (Linear) ---
@st.cache_resource
def load_fast_model():
    data = [
        ("I am confident in my skills", "self-confident"),
        ("I am unsure and lost", "uncertain"),
        ("I love learning and growing", "growth-oriented"),
        ("I analyze the data strictly", "analytical"),
        ("I create art and stories", "creative"),
        ("I am skeptical of new claims", "skeptical"),
        ("I avoid all risks", "risk-averse")
    ]
    texts, labels = zip(*data)
    model = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])
    model.fit(texts, labels)
    return model

# --- Model 2 & 3: High-Accuracy (Transformers) ---
@st.cache_resource
def load_transformer_model(model_name):
    return pipeline("zero-shot-classification", model=model_name)

# --- UI Setup ---
st.title("👤 Identity & Belief Evaluation Suite")

default_text = (
    "I've been reflecting on my career lately. I used to be very risk-averse, "
    "but lately, I find myself excited by the challenge of building new things from scratch. "
    "I'm becoming much more of a growth-oriented leader than I was a year ago."
)

with st.sidebar:
    st.header("Configuration")
    model_choice = st.radio(
        "Select Model Engine:",
        (
            "High Speed (Naive Bayes)", 
            "High Accuracy (DistilBART)", 
            "High Accuracy (BART-Large)"
        )
    )
    
    st.markdown("---")
    labels_input = st.text_input(
        "Analysis Labels", 
        "self-confident, uncertain, growth-oriented, analytical, creative, skeptical, risk-averse"
    )
    candidate_labels = [label.strip() for label in labels_input.split(",")]
    threshold = st.slider("Min Confidence Score", 0.0, 1.0, 0.2)

# --- Main Input ---
user_input = st.text_area("Conversation Input:", value=default_text, height=150)
analyze_button = st.button("Analyze Identity Signals", type="primary")

# --- Execution Logic ---
if analyze_button and user_input:
    start_time = time.time()
    
    if model_choice == "High Speed (Naive Bayes)":
        model = load_fast_model()
        probs = model.predict_proba([user_input])[0]
        classes = model.classes_
        raw_res = {label: float(prob) for label, prob in zip(classes, probs)}
        formatted_signals = [
            {"Signal": label, "Confidence": round(prob, 4)}
            for label, prob in raw_res.items() if prob >= threshold
        ]
    else:
        # Determine which HF model to load
        hf_model_path = "valhalla/distilbart-mnli-12-3" if model_choice == "High Accuracy (DistilBART)" else "facebook/bart-large-mnli"
        
        with st.spinner(f"Running {model_choice}..."):
            model = load_transformer_model(hf_model_path)
            res = model(user_input, candidate_labels, multi_label=True)
            raw_res = res
            formatted_signals = [
                {"Signal": label, "Confidence": round(score, 4)}
                for label, score in zip(res['labels'], res['scores']) if score >= threshold
            ]
    
    latency_ms = (time.time() - start_time) * 1000
    st.session_state['eval_result'] = {
        "signals": formatted_signals,
        "raw": raw_res,
        "latency": latency_ms,
        "engine": model_choice
    }

# --- Results & Visualization ---
if 'eval_result' in st.session_state:
    res = st.session_state['eval_result']
    
    st.metric("Inference Latency", f"{res['latency']:.2f} ms", 
              delta="Target: <200ms", delta_color="normal" if res['latency'] < 200 else "inverse")

    if res['signals']:
        df = pd.DataFrame(res['signals']).sort_values("Confidence", ascending=True)
        fig = px.bar(df, x="Confidence", y="Signal", orientation='h', 
                     color="Confidence", color_continuous_scale="Viridis", text_auto='.2f')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Data Export")
        col1, col2 = st.columns([1, 1])
        
        json_output = json.dumps(res['raw'], indent=2)
        
        with col1:
            st.markdown("**Visible JSON Output:**")
            st.code(json_output, language="json")
        
        with col2:
            st.markdown("**Download Options:**")
            st.download_button(
                label="📥 Download JSON File",
                data=json_output,
                file_name=f"identity_{int(time.time())}.json",
                mime="application/json"
            )
            st.info(f"Engine used: {res['engine']}")
    else:
        st.warning("No signals met the confidence threshold.")import streamlit as st
import pandas as pd
import plotly.express as px
import json
import time
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# --- Page Configuration ---
st.set_page_config(page_title="Identity Evaluation Suite", layout="wide")

# --- Model 1: High-Speed (Linear) ---
@st.cache_resource
def load_fast_model():
    data = [
        ("I am confident in my skills", "self-confident"),
        ("I am unsure and lost", "uncertain"),
        ("I love learning and growing", "growth-oriented"),
        ("I analyze the data strictly", "analytical"),
        ("I create art and stories", "creative"),
        ("I am skeptical of new claims", "skeptical"),
        ("I avoid all risks", "risk-averse")
    ]
    texts, labels = zip(*data)
    model = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])
    model.fit(texts, labels)
    return model

# --- Model 2 & 3: High-Accuracy (Transformers) ---
@st.cache_resource
def load_transformer_model(model_name):
    return pipeline("zero-shot-classification", model=model_name)

# --- UI Setup ---
st.title("👤 Identity & Belief Evaluation Suite")

default_text = (
    "I've been reflecting on my career lately. I used to be very risk-averse, "
    "but lately, I find myself excited by the challenge of building new things from scratch. "
    "I'm becoming much more of a growth-oriented leader than I was a year ago."
)

with st.sidebar:
    st.header("Configuration")
    model_choice = st.radio(
        "Select Model Engine:",
        (
            "High Speed (Naive Bayes)", 
            "High Accuracy (DistilBART)", 
            "High Accuracy (BART-Large)"
        )
    )
    
    st.markdown("---")
    labels_input = st.text_input(
        "Analysis Labels", 
        "self-confident, uncertain, growth-oriented, analytical, creative, skeptical, risk-averse"
    )
    candidate_labels = [label.strip() for label in labels_input.split(",")]
    threshold = st.slider("Min Confidence Score", 0.0, 1.0, 0.2)

# --- Main Input ---
user_input = st.text_area("Conversation Input:", value=default_text, height=150)
analyze_button = st.button("Analyze Identity Signals", type="primary")

# --- Execution Logic ---
if analyze_button and user_input:
    start_time = time.time()
    
    if model_choice == "High Speed (Naive Bayes)":
        model = load_fast_model()
        probs = model.predict_proba([user_input])[0]
        classes = model.classes_
        raw_res = {label: float(prob) for label, prob in zip(classes, probs)}
        formatted_signals = [
            {"Signal": label, "Confidence": round(prob, 4)}
            for label, prob in raw_res.items() if prob >= threshold
        ]
    else:
        # Determine which HF model to load
        hf_model_path = "valhalla/distilbart-mnli-12-3" if model_choice == "High Accuracy (DistilBART)" else "facebook/bart-large-mnli"
        
        with st.spinner(f"Running {model_choice}..."):
            model = load_transformer_model(hf_model_path)
            res = model(user_input, candidate_labels, multi_label=True)
            raw_res = res
            formatted_signals = [
                {"Signal": label, "Confidence": round(score, 4)}
                for label, score in zip(res['labels'], res['scores']) if score >= threshold
            ]
    
    latency_ms = (time.time() - start_time) * 1000
    st.session_state['eval_result'] = {
        "signals": formatted_signals,
        "raw": raw_res,
        "latency": latency_ms,
        "engine": model_choice
    }

# --- Results & Visualization ---
if 'eval_result' in st.session_state:
    res = st.session_state['eval_result']
    
    st.metric("Inference Latency", f"{res['latency']:.2f} ms", 
              delta="Target: <200ms", delta_color="normal" if res['latency'] < 200 else "inverse")

    if res['signals']:
        df = pd.DataFrame(res['signals']).sort_values("Confidence", ascending=True)
        fig = px.bar(df, x="Confidence", y="Signal", orientation='h', 
                     color="Confidence", color_continuous_scale="Viridis", text_auto='.2f')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Data Export")
        col1, col2 = st.columns([1, 1])
        
        json_output = json.dumps(res['raw'], indent=2)
        
        with col1:
            st.markdown("**Visible JSON Output:**")
            st.code(json_output, language="json")
        
        with col2:
            st.markdown("**Download Options:**")
            st.download_button(
                label="📥 Download JSON File",
                data=json_output,
                file_name=f"identity_{int(time.time())}.json",
                mime="application/json"
            )
            st.info(f"Engine used: {res['engine']}")
    else:
        st.warning("No signals met the confidence threshold.")
