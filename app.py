import streamlit as st
import librosa
import numpy as np
import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt

from feature_extraction import extract_features, transcribe_audio, record_audio
from sentiment import analyze_sentiment

# Set page configuration for a premium look
st.set_page_config(
    page_title="AI Deception Detector",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for advanced UI aesthetics
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4da6ff;
    }
    
    /* Card-like containers for results */
    .result-card {
        background-color: #1a1c23;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 24px;
        border: 1px solid #2d3139;
        transition: transform 0.2s ease-in-out;
    }
    .result-card:hover {
        transform: translateY(-2px);
    }
    
    /* Progress bar enhancements */
    .stProgress > div > div > div > div {
        background-color: #4da6ff;
        background-image: linear-gradient(90deg, #4da6ff, #9933ff);
    }
    
    /* Warning/Info boxes */
    .stAlert {
        border-radius: 8px;
        border: none;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #12141a;
        border-right: 1px solid #2d3139;
    }
    
    /* Prediction text */
    .truth-text {
        color: #00e676;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(0, 230, 118, 0.3);
    }
    .lie-text {
        color: #ff1744;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(255, 23, 68, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    model_path = 'models/lie_detector_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# Header Section
st.title("🕵️‍♂️ AI Deception Detection System")
st.markdown("""
    <div style='background-color: #1a1c23; padding: 15px; border-radius: 8px; border-left: 5px solid #4da6ff; margin-bottom: 30px;'>
        <h4 style='margin:0; color: #4da6ff;'>Advanced Multi-Modal Analysis</h4>
        <p style='margin: 10px 0 0 0; color: #a0aabf; font-size: 14px;'>
            This system analyzes acoustic vocal patterns alongside NLP semantic sentiment to estimate the probability of stress-induced deception.
        </p>
    </div>
""", unsafe_allow_html=True)

def plot_waveform(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.set_facecolor('#1a1c23')
        fig.patch.set_facecolor('#1a1c23')
        
        # Plot styling
        ax.plot(np.linspace(0, len(y)/sr, len(y)), y, color='#4da6ff', alpha=0.8, linewidth=1)
        ax.set_title("Audio Waveform Signature", color='white', pad=10)
        ax.axis('off')
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error("Could not generate waveform.")

# Function to run inference
def run_analysis(file_path):
    # Create main layout columns
    col1, col2 = st.columns([1.2, 1])
    
    with st.spinner("Processing intelligence... Analyzing voice signatures and sentiment..."):
        # 1. Transcribe audio
        transcript = transcribe_audio(file_path)
        
        # 2. Sentiment Analysis
        sentiment_score = analyze_sentiment(transcript) if transcript else 0.0
        
        # 3. Voice Feature Extraction
        audio_features = extract_features(file_path, transcript)
        
        # 4. Model Prediction
        prob_lie = 0.0
        prob_truth = 1.0
        prediction_ready = False
        
        if model is not None and transcript:
            feature_vector = np.array([audio_features + [sentiment_score]])
            prob_truth = model.predict_proba(feature_vector)[0][0]
            prob_lie = model.predict_proba(feature_vector)[0][1]
            prediction_ready = True
            
    # Waveform display at the very top
    with st.spinner("⏳ Rendering Acoustic Waveform Signature..."):
        plot_waveform(file_path)
        
    st.markdown("---")

    # DISPLAY RESULTS
    with col1:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.subheader("Transcription & NLP Analysis")
        if not transcript:
            st.warning("No speech detected or transcription failed. (Ensure 'ffmpeg' is installed on your Mac!)")
            transcript_display = "[No intelligible speech found]"
        else:
            transcript_display = transcript
            
        st.info(f"**Detected Speech:**\n\n\"{transcript_display}\"")
        
        st.write("---")
        st.write("**Semantic Sentiment Analysis (Graph)**")
        
        # Bar graph for Sentiment
        fig_sent, ax_sent = plt.subplots(figsize=(5, 1))
        ax_sent.set_facecolor('#1a1c23')
        fig_sent.patch.set_facecolor('#1a1c23')
        
        bar_color = '#00e676' if sentiment_score > 0 else '#ff1744'
        ax_sent.barh(['Sentiment'], [sentiment_score], color=bar_color, height=0.4)
        ax_sent.set_xlim(-1.1, 1.1)
        ax_sent.axvline(0, color='white', linewidth=1, linestyle='--')
        ax_sent.tick_params(colors='white', bottom=False, labelbottom=True)
        ax_sent.spines['top'].set_visible(False)
        ax_sent.spines['right'].set_visible(False)
        ax_sent.spines['bottom'].set_visible(False)
        ax_sent.spines['left'].set_visible(False)
        
        st.pyplot(fig_sent)
        
        sentiment_label = "Neutral"
        if sentiment_score > 0.2: sentiment_label = "Positive 😊"
        elif sentiment_score < -0.2: sentiment_label = "Negative 😠 / Stressed"
        
        st.caption(f"Score: {sentiment_score:.2f} ({sentiment_label})")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.subheader(" Acoustic Features")
        
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("MFCC (Mean)", f"{audio_features[0]:.2f}")
        m_col2.metric("Pitch Var", f"{audio_features[1]:.2f}")
        
        m_col3, m_col4 = st.columns(2)
        m_col3.metric("Energy Mean", f"{audio_features[2]:.4f}")
        m_col4.metric("Speech Rate", f"{audio_features[3]:.1f} wpm")
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.markdown("---")
    
    # PREDICTION DASHBOARD
    st.markdown("<h2 style='text-align: center; margin-bottom: 20px;'>Final Verdict Analysis</h2>", unsafe_allow_html=True)
    
    if not prediction_ready:
        st.error("Incomplete data pipeline. Cannot render verdict.")
        return
        
    d_col1, d_col2 = st.columns([1, 2])
    
    with d_col1:
        st.metric(label="Deception Probability", value=f"{prob_lie * 100:.1f}%", delta=f"{(prob_lie - prob_truth)*100:.1f}% certainty", delta_color="inverse")
    
    with d_col2:
        st.write("**Deception Confidence Meter**")
        st.progress(prob_lie)
        
        prediction_text = "TRUTHFUL" if prob_truth > prob_lie else "DECEPTIVE"
        css_class = "truth-text" if prediction_text == "TRUTHFUL" else "lie-text"
        
        st.markdown(f"""
            <div style='text-align: center; padding: 20px; background-color: #1a1c23; border-radius: 8px; margin-top: 15px;'>
                <h3 style='margin: 0;'>System suggests the subject is:</h3>
                <h1 class='{css_class}' style='font-size: 3rem; margin: 10px 0;'>{prediction_text}</h1>
            </div>
        """, unsafe_allow_html=True)

# Sidebar setup
st.sidebar.markdown("<h3 style='text-align: center;'>Control Panel</h3>", unsafe_allow_html=True)
st.sidebar.markdown("---")
option = st.sidebar.radio("Choose Input Method:", ("Upload Audio", "Record Live Audio"))
st.sidebar.markdown("---")

if model is None:
    st.sidebar.error(" Model payload `lie_detector_model.pkl` not found! Features available, but final probability will not map.")

if option == "Upload Audio":
    st.subheader(" Upload Audio Signature")
    st.write("Upload a target WAV or MP3 file for deep analysis.")
    uploaded_file = st.file_uploader("Drop file here...", type=["wav", "mp3", "m4a"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        file_path = f"audio_samples/temp_{uploaded_file.name}"
        os.makedirs("audio_samples", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.audio(file_path)
        
        # Centered button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 Execute Advanced Analysis", use_container_width=True):
                run_analysis(file_path)

elif option == "Record Live Audio":
    st.subheader("🎙️ Live Interrogation Mode")
    st.markdown("<p style='color: #a0aabf;'>Ensure your microphone is connected and authorized. Recording will commence immediately.</p>", unsafe_allow_html=True)
    
    duration = st.slider("Select Recording Duration (seconds)", 3, 15, 5)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(f"🔴 Start {duration}s Recording", use_container_width=True):
            file_path = "audio_samples/live_recording.wav"
            os.makedirs("audio_samples", exist_ok=True)
            
            # Recording animation placeholder
            status_placeholder = st.empty()
            with status_placeholder.container():
                st.markdown("""
                    <div style='text-align:center;'>
                        <h2 style='color:#ff1744;'>🔴 RECORDING IN PROGRESS...</h2>
                        <img src="https://i.gifer.com/origin/b4/b4d657e7ef262b88eb5f7ac021edda87.gif" width="100"/>
                        <p style='color:#a0aabf; font-style:italic;'>Speak clearly into the microphone...</p>
                    </div>
                """, unsafe_allow_html=True)
                
            try:
                record_audio(file_path, duration=duration)
                status_placeholder.empty() # Clear animation
                st.success(" Audio captured successfully. Beginning intelligence extraction...")
                st.audio(file_path)
                
                run_analysis(file_path)
            except Exception as e:
                status_placeholder.empty()
                st.error(f"Microphone critical failure: {e}")
