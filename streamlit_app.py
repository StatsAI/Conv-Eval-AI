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
def run_inference(text, engine_choice, labels, threshold):
    if engine_choice == "High Speed (Naive Bayes)":
        model = load_fast_model()
        probs = model.predict_proba([text])[0]
        classes = model.classes_
        return [{"Signal": l, "Confidence": round(p, 4)} for l, p in zip(classes, probs) if p >= threshold]
    else:
        path = "valhalla/distilbart-mnli-12-3" if "DistilBART" in engine_choice else "facebook/bart-large-mnli"
        model = load_transformer_model(path)
        res = model(text, labels, multi_label=True)
        return [{"Signal": l, "Confidence": round(s, 4)} for l, s in zip(res['labels'], res['scores']) if s >= threshold]

# --- UI Sidebar ---
with st.sidebar:
    st.header("Configuration")
    app_mode = st.radio("Select Mode:", ("Interactive Mode", "JSON Mode"))
    model_choice = st.radio("Select Model Engine:", ("High Speed (Naive Bayes)", "High Accuracy (DistilBART)", "High Accuracy (BART-Large)"))
    labels_input = st.text_input("Analysis Labels", "self-confident, uncertain, growth-oriented, analytical, creative, skeptical, risk-averse")
    candidate_labels = [label.strip() for label in labels_input.split(",")]
    threshold = st.slider("Min Confidence Score", 0.0, 1.0, 0.2)

st.title(f"👤 Identity Evaluation: {app_mode}")

# --- INTERACTIVE MODE ---
if app_mode == "Interactive Mode":
    user_input = st.text_area("Conversation Input:", value="I'm feeling much more of a growth-oriented leader than I was a year ago.", height=150)
    if st.button("Analyze Identity Signals", type="primary"):
        start = time.time()
        signals = run_inference(user_input, model_choice, candidate_labels, threshold)
        latency = (time.time() - start) * 1000
        
        st.metric("Latency", f"{latency:.2f} ms")
        if signals:
            df = pd.DataFrame(signals).sort_values("Confidence", ascending=True)
            st.plotly_chart(px.bar(df, x="Confidence", y="Signal", orientation='h', color="Confidence"), use_container_width=True)
            st.code(json.dumps(signals, indent=2), language="json")
            st.download_button("Download JSON", json.dumps(signals), "identity.json")

# --- JSON MODE ---
else:
    uploaded_file = st.file_uploader("Upload StoryBot Chat Log (JSON)", type="json")
    if uploaded_file is not None:
        chat_log = json.load(uploaded_file)
        results = []
        
        if st.button("Process Chat Log", type="primary"):
            progress_bar = st.progress(0)
            for i, entry in enumerate(chat_log):
                # Only analyze User messages
                if entry.get("sender") == "User":
                    signals = run_inference(entry["text"], model_choice, candidate_labels, threshold)
                    for s in signals:
                        results.append({
                            "Timestamp": f"{entry['metadata']['date']} {entry['metadata']['time']}",
                            "Signal": s["Signal"],
                            "Confidence": s["Confidence"],
                            "User": entry['metadata']['userid']
                        })
                progress_bar.progress((i + 1) / len(chat_log))

            if results:
                df_results = pd.DataFrame(results)
                
                # Time Series Visualization
                st.subheader("Signal Evolution Over Time")
                fig = px.line(df_results, x="Timestamp", y="Confidence", color="Signal", markers=True,
                             title=f"Identity Evolution for User: {results[0]['User']}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Data Export
                st.subheader("Session Analysis Output")
                final_json = df_results.to_json(orient="records")
                st.code(final_json, language="json")
                st.download_button("Download Session Signals", final_json, "session_analysis.json")
            else:
                st.warning("No user signals detected in the log.")
