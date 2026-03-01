import os
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

from feature_extraction import extract_features, transcribe_audio
from sentiment import analyze_sentiment

def load_data(csv_file="data/labels.csv"):
    X, y = [], []
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            file_path = row['file']
            label = int(row['label'])
            
            print(f"Processing {file_path}...")
            # 1. Transcribe audio
            transcript = transcribe_audio(file_path)
            
            # 2. Extract sentiment from text
            sentiment_score = analyze_sentiment(transcript)
            
            # 3. Extract audio features
            audio_features = extract_features(file_path, transcript)
            
            # 4. Combine into single feature vector
            # [mfcc, pitch_var, energy_mean, speech_rate, sentiment_score]
            feature_vector = audio_features + [sentiment_score]
            
            X.append(feature_vector)
            y.append(label)
            
    return np.array(X), np.array(y)

if __name__ == "__main__":
    print("Loading dataset and extracting features. This might take some time...")
    X, y = load_data()
    
    if len(X) == 0:
        print("No data to train on!")
        exit(1)
        
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    # Train-test split (handle small dummy datasets gracefully)
    test_size = 0.2 if len(X) > 5 else 0.5
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    print("Training RandomForest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    os.makedirs('models', exist_ok=True)
    model_path = 'models/lie_detector_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
