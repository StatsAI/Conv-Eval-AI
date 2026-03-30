import streamlit as st
import pandas as pd
import plotly.express as px
import json
import time
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Identity Evaluation Suite", layout="wide")

# --- Model Loaders ---
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

@st.cache_resource
def load_transformer_model(model_name):
    return pipeline("zero-shot-classification", model=model_name)

# --- Logic Helper ---
def run_full_inference(text, engine_choice, labels):
    """Returns scores for ALL labels regardless of threshold for data integrity."""
    if engine_choice == "High Speed (Naive Bayes)":
        model = load_fast_model()
        probs = model.predict_proba([text])[0]
        classes = model.classes_
        return {l: round(float(p), 4) for l, p in zip(classes, probs)}
    else:
        path = "valhalla/distilbart-mnli-12-3" if "DistilBART" in engine_choice else "facebook/bart-large-mnli"
        model = load_transformer_model(path)
        res = model(text, labels, multi_label=True)
        return {l: round(float(s), 4) for l, s in zip(res['labels'], res['scores'])}

# --- UI Sidebar ---
with st.sidebar:
    st.header("Configuration")
    app_mode = st.radio("Select Mode:", ("Interactive Mode", "JSON Mode"))
    model_choice = st.radio("Select Model Engine:", ("High Speed (Naive Bayes)", "High Accuracy (DistilBART)", "High Accuracy (BART-Large)"))
    labels_input = st.text_input("Analysis Labels", "self-confident, uncertain, growth-oriented, analytical, creative, skeptical, risk-averse")
    candidate_labels = [label.strip() for label in labels_input.split(",")]
    threshold = st.slider("Min Confidence Score (Visual Filter Only)", 0.0, 1.0, 0.2)

st.title(f"👤 Identity Evaluation: {app_mode}")

# --- INTERACTIVE MODE ---
if app_mode == "Interactive Mode":
    user_input = st.text_area("Conversation Input:", value="I am becoming much more of a growth-oriented leader.", height=150)
    if st.button("Analyze Identity Signals", type="primary"):
        start = time.time()
        all_scores = run_full_inference(user_input, model_choice, candidate_labels)
        latency = (time.time() - start) * 1000
        
        st.metric("Latency", f"{latency:.2f} ms")
        df = pd.DataFrame(list(all_scores.items()), columns=["Signal", "Confidence"])
        filtered_df = df[df["Confidence"] >= threshold].sort_values("Confidence", ascending=True)
        
        st.plotly_chart(px.bar(filtered_df, x="Confidence", y="Signal", orientation='h', color="Confidence"), use_container_width=True)
        st.code(json.dumps(all_scores, indent=2), language="json")

# --- JSON MODE ---
else:
    uploaded_file = st.file_uploader("Upload StoryBot Chat Log (JSON)", type="json")
    if uploaded_file is not None:
        input_log = json.load(uploaded_file)
        
        if st.button("Enrich Chat Log", type="primary"):
            enriched_log = []
            time_series_data = []
            progress_bar = st.progress(0)
            
            for i, entry in enumerate(input_log):
                # Create a copy of the original entry
                enriched_entry = entry.copy()
                
                # Analyze if sender is User
                if entry.get("sender") == "User":
                    scores = run_full_inference(entry["text"], model_choice, candidate_labels)
                    enriched_entry["identity_scores"] = scores
                    
                    # Prepare data for the time series chart
                    timestamp = f"{entry['metadata']['date']} {entry['metadata']['time']}"
                    for label, score in scores.items():
                        time_series_data.append({
                            "Timestamp": timestamp,
                            "Signal": label,
                            "Confidence": score
                        })
                else:
                    # For StoryBot, we append empty scores or null to keep schema consistent
                    enriched_entry["identity_scores"] = None
                
                enriched_log.append(enriched_entry)
                progress_bar.progress((i + 1) / len(input_log))

            # --- Visualizations ---
            if time_series_data:
                df_plot = pd.DataFrame(time_series_data)
                st.subheader("Identity Evolution Time Series")
                # Filter plot by threshold to keep chart clean
                fig = px.line(df_plot[df_plot["Confidence"] >= threshold], 
                             x="Timestamp", y="Confidence", color="Signal", markers=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # --- Output ---
                st.subheader("Enriched JSON Output")
                final_json_str = json.dumps(enriched_log, indent=2)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.code(final_json_str, language="json")
                with col2:
                    st.download_button(
                        label="📥 Download Enriched Log",
                        data=final_json_str,
                        file_name="enriched_storybot_log.json",
                        mime="application/json"
                    )
            else:
                st.warning("No user messages found to analyze.")
