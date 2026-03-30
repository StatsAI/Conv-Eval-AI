import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px
import json

# --- Page Configuration ---
st.set_page_config(page_title="Identity Signal Evaluator", layout="wide")

# --- Model Loading (Cached) ---
@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_classifier()

# --- App UI ---
st.title("👤 Identity & Belief Evaluation")
st.markdown("""
This tool evaluates conversational data to identify evolving user beliefs and self-perceptions. 
""")

# --- Default Text ---
default_text = (
    "I've been reflecting on my career lately. I used to be very risk-averse, "
    "but lately, I find myself excited by the challenge of building new things from scratch. "
    "I'm becoming much more of a growth-oriented leader than I was a year ago."
)

with st.sidebar:
    st.header("Settings")
    labels_input = st.text_input(
        "Analysis Labels (comma separated)", 
        "self-confident, uncertain, growth-oriented, analytical, creative, skeptical, risk-averse"
    )
    candidate_labels = [label.strip() for label in labels_input.split(",")]
    threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.4)

# --- Input Section ---
user_input = st.text_area(
    "Conversation Input:", 
    value=default_text,
    height=150
)

analyze_button = st.button("Analyze Identity Signals", type="primary")

# --- Analysis Logic ---
if analyze_button and user_input:
    with st.spinner("Analyzing text..."):
        # Run inference
        result = classifier(user_input, candidate_labels, multi_label=True)
        
        # Process results
        data = [
            {"Signal": label, "Confidence": round(score, 4)}
            for label, score in zip(result['labels'], result['scores'])
            if score >= threshold
        ]
        
        if data:
            df = pd.DataFrame(data)
            
            # 1. Visualization below the button
            st.subheader("Identity Signal Distribution")
            fig = px.bar(
                df, 
                x="Confidence", 
                y="Signal", 
                orientation='h',
                color="Confidence",
                color_continuous_scale="Viridis",
                title="Consolidated Identity Profile"
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # 2. Results Table
            st.subheader("Extracted Signals Data")
            st.dataframe(df, use_container_width=True)
            
            # 3. JSON Download
            st.subheader("Developer Export")
            json_string = json.dumps(result, indent=2)
            st.download_button(
                label="Download Analysis JSON",
                file_name="identity_evaluation.json",
                mime="application/json",
                data=json_string,
            )
            
            st.success("Analysis Complete.")
        else:
            st.warning("No signals met the confidence threshold.")

elif analyze_button and not user_input:
    st.error("Please enter some text to analyze.")
