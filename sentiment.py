from transformers import pipeline

sentiment_model = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    if not text.strip():
        return 0.0 # Neutral if empty
    
    result = sentiment_model(text[:512])[0] # Limit to 512 chars for model constraints
    label = result["label"]
    score = result["score"]
    
    # Convert to a continuous range [-1, 1]
    if label == "POSITIVE":
        return score
    elif label == "NEGATIVE":
        return -score
    else:
        return 0.0