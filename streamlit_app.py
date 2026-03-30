import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(page_title="Identity Signal Evaluator", layout="wide")

# --- Model Loading (Cached) ---
@st.cache_resource
def load_classifier():
    # Loading the BART model for Zero-Shot Classification
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_classifier()

# --- App UI ---
st.title("👤 Identity & Belief Evaluation")
st.markdown("""
This tool evaluates conversational data to identify evolving user beliefs and self-perceptions. 
Enter a user message below to see how the system extracts **Identity Signals**.
""")

with st.sidebar:
    st.header("Settings")
    # Allow teams to define what "labels" they are looking for
    labels_input = st.text_input(
        "Analysis Labels (comma separated)", 
        "self-confident, uncertain, growth-oriented, analytical, creative, skeptical"
    )
    candidate_labels = [label.strip() for label in labels_input.split(",")]
    
    threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.4)

# --- Input Section ---
col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area(
        "Conversation Input:", 
        placeholder="Example: I've started taking more risks with my projects lately. I used to be afraid of failure, but now I see it as a learning step.",
        height=200
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
            
            with col1:
                st.subheader("Detected Signals")
                st.dataframe(df, use_container_width=True)
            
            with col2:
                st.subheader("Visual Distribution")
                fig = px.bar(
                    df, 
                    x="Confidence", 
                    y="Signal", 
                    orientation='h',
                    color="Confidence",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            st.success("Analysis Complete. These signals can now be sent to Recommendation or StoryBot engines.")
        else:
            st.warning("No signals met the confidence threshold. Try adjusting the threshold in the sidebar.")

elif analyze_button and not user_input:
    st.error("Please enter some text to analyze.")

# --- Metadata/Developer View ---
with st.expander("View Raw JSON Output (For API Teams)"):
    if 'result' in locals():
        st.json(result)
    else:
        st.write("Run an analysis to see the JSON schema.")
