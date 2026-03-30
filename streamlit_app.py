import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px
import json

# --- Page Configuration ---
st.set_page_config(page_title="Identity Signal Evaluator", layout="wide")

# --- Optimized Model Loading ---
@st.cache_resource
def load_classifier():
    # Swapping to distilbart for ~2x speed improvement on CPU
    return pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3")

classifier = load_classifier()

# --- App UI ---
st.title("👤 Identity & Belief Evaluation")

# Default text for immediate testing
default_text = (
    "I've been reflecting on my career lately. I used to be very risk-averse, "
    "but lately, I find myself excited by the challenge of building new things from scratch. "
    "I'm becoming much more of a growth-oriented leader than I was a year ago."
)

with st.sidebar:
    st.header("Settings")
    labels_input = st.text_input(
        "Analysis Labels", 
        "self-confident, uncertain, growth-oriented, analytical, creative, skeptical, risk-averse"
    )
    candidate_labels = [label.strip() for label in labels_input.split(",")]
    threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.4)

# --- Input Section ---
user_input = st.text_area("Conversation Input:", value=default_text, height=150)
analyze_button = st.button("Analyze Identity Signals", type="primary")

# --- Persistent Analysis Logic ---
if analyze_button and user_input:
    with st.spinner("Analyzing... (Using Optimized DistilBART)"):
        # Run inference
        result = classifier(user_input, candidate_labels, multi_label=True)
        st.session_state['last_result'] = result

# --- Visualization & Export (Below Button) ---
if 'last_result' in st.session_state:
    res = st.session_state['last_result']
    
    # Filter by threshold
    filtered_data = [
        {"Signal": label, "Confidence": round(score, 4)}
        for label, score in zip(res['labels'], res['scores'])
        if score >= threshold
    ]
    
    if filtered_data:
        df = pd.DataFrame(filtered_data)
        
        # Consolidated Chart
        st.subheader("Identity Signal Distribution")
        fig = px.bar(
            df, 
            x="Confidence", 
            y="Signal", 
            orientation='h',
            color="Confidence",
            color_continuous_scale="Blugrn",
            text_auto='.2f'
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Download Section
        col1, col2 = st.columns([1, 4])
        with col1:
            json_string = json.dumps(res, indent=2)
            st.download_button(
                label="📥 Download JSON",
                file_name="identity_evaluation.json",
                mime="application/json",
                data=json_string,
            )
        with col2:
            st.info("Analysis complete. Signals extracted for StoryBot personalization.")
    else:
        st.warning("No signals met the current threshold. Try lowering it in the sidebar.")
