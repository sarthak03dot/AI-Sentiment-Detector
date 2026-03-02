# 🎤 AI-Based Deception Detection System
## (Voice + Sentiment + Stress Analysis)

---

# 📌 1. Project Title
Stress-Based Deception Prediction System using Voice and NLP

---

# 📌 2. Abstract
This project aims to build an AI-based system that predicts the likelihood of deception using voice acoustic features and sentiment analysis.

The system:
- Records user voice
- Converts speech to text
- Extracts voice features (MFCC, pitch, energy)
- Performs sentiment analysis
- Combines all features
- Predicts Lie Probability Score

⚠ Note: This is stress-based prediction. It does NOT guarantee 100% lie detection.

---

# 📌 3. Problem Statement
Traditional polygraph systems:
- Expensive
- Hardware dependent
- Not easily accessible

This project provides:
- Low-cost AI solution
- Fully software-based approach
- Free & open-source implementation

---

# 📌 4. System Architecture
User Voice Input
↓
Speech-to-Text (Whisper)
↓
Voice Feature Extraction (Librosa)
↓
Sentiment Analysis (Transformers)
↓
Feature Combination
↓
Machine Learning Model (RandomForest)
↓
Lie Probability Output

---

# 📌 5. Technologies Used (100% Free)
Programming Language:
- Python 3.11

Speech-to-Text:
- Whisper (Open Source)

Audio Processing:
- Librosa

Machine Learning:
- Scikit-learn (RandomForest)

Sentiment Analysis:
- HuggingFace Transformers

Frontend:
- Streamlit

Environment:
- pyenv + venv

Deployment:
- Streamlit Cloud (Free)

---

# 📌 6. Multi-Platform Setup Guide (Windows & macOS)

### 🛠️ Mandatory Prerequisites (Before you start)
For both Windows and Mac, you **MUST** install `ffmpeg` for audio processing and Whisper transcription to work.

- **Windows**: 
  1. Download from [ffmpeg.org](https://ffmpeg.org/download.html).
  2. Add the `ffmpeg/bin` folder to your System ENV Path.
- **Mac**: 
  1. `brew install ffmpeg`

---

## 💻 Windows Setup (Manual or pyenv)

### 1. Clone the repository
```bash
git clone https://github.com/your-username/AI-Lie-Detector.git
cd AI-Lie-Detector
```

### 2. Create and activate a Virtual Environment
```powershell
python -m venv lie_env
.\lie_env\Scripts\activate
```

### 3. Install dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🍎 macOS Setup (brew + pyenv)

### 1. Install pyenv and dependencies
```bash
brew install pyenv openssl readline sqlite3 xz zlib tcl-tk
```

### 2. Install Python 3.11.8
```bash
env PYTHON_CONFIGURE_OPTS="--without-tk" pyenv install 3.11.8
pyenv local 3.11.8
```

### 3. Create and activate a Virtual Environment
```bash
python -m venv lie_env
source lie_env/bin/activate
```

### 4. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 7. Running the Application

After activating your virtual environment, run:

```bash
streamlit run app.py
```

---

# 📌 7. Project Folder Structure
AI-Lie-Detector/
│
├── lie_env/
├── data/
├── models/
├── audio_samples/
├── feature_extraction.py
├── sentiment.py
├── train_model.py
├── app.py
├── requirements.txt
└── README.md

---

# 📌 8. Modules Description

## 🔹 Module 1: Voice Recording
- Records microphone input
- Saves audio as WAV file

## 🔹 Module 2: Speech-to-Text
- Uses Whisper model
- Converts audio to text

## 🔹 Module 3: Voice Feature Extraction
Extract:
- MFCC
- Pitch
- Energy
- Zero Crossing Rate

## 🔹 Module 4: Sentiment Analysis
- Uses pretrained transformer model
- Extracts positive/negative scores

## 🔹 Module 5: Feature Engineering
Combine:
[MFCC_mean, Pitch_var, Energy_mean, Sentiment_score, Speech_rate]

## 🔹 Module 6: Machine Learning Model
Algorithm:
- RandomForest Classifier

Output:
0 = Truth
1 = Lie

---

# 📌 9. Dataset

Option 1:
- RAVDESS (Emotion dataset)
- CREMA-D

Option 2:
Custom dataset:
- Record truthful answers
- Record fake answers
- Label manually

---

# 📌 10. Model Training Steps
1. Collect dataset
2. Extract features
3. Prepare feature vector
4. Split train/test
5. Train RandomForest
6. Evaluate accuracy
7. Save model (.pkl file)

---

# 📌 11. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

# 📌 12. Limitations
- Stress ≠ Lie always
- Background noise affects accuracy
- Cultural speech variation
- Small dataset reduces reliability
