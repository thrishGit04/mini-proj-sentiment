from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import datetime
import os
import sys

# To support importing from local inference.py
sys.path.append(os.path.dirname(__file__))

app = FastAPI(title="Sentiment Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    text: str

# Globals for metrics
total_analyses = 0
positive_count = 0
negative_count = 0
neutral_count = 0

# Try to load inference model
inference = None
try:
    from inference import SentimentInference
    # Adjust model dir based on cwd
    model_path = os.path.join(os.path.dirname(__file__), "saved_model")
    if os.path.exists(model_path):
        inference = SentimentInference(model_dir=model_path)
    else:
        print("Warning: saved_model not found. Train the model first.")
except Exception as e:
    print(f"Failed to load model: {e}")

# In-memory history for /history endpoint if needed
analysis_history = []

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    global total_analyses, positive_count, negative_count, neutral_count
    
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="No text provided")
        
    try:
        if not inference:
            # Smart Rule-Based (Keyword) fallback sentiment analysis
            text_lower = req.text.lower()
            
            # Define word lists
            pos_words = ["love", "loved", "like", "liked", "good", "great", "excellent", "amazing", "wonderful", "fantastic", "awesome", "perfect", "beautiful", "best", "enjoy", "enjoyed"]
            neg_words = ["hate", "hated", "dislike", "bad", "terrible", "worst", "awful", "waste", "boring", "poor", "disappointed", "crap", "rubbish", "horrible", "annoying", "fail"]
            
            pos_score = sum(1 for word in pos_words if word in text_lower)
            neg_score = sum(1 for word in neg_words if word in text_lower)
            
            # Simple rule-based logic
            if pos_score > neg_score:
                mock_sentiment = "Positive"
                confidence = min(0.75 + 0.05 * (pos_score - neg_score), 0.98)
                p_pos_raw = confidence
                p_neg_raw = 0.05 + 0.02 * neg_score
            elif neg_score > pos_score:
                mock_sentiment = "Negative"
                confidence = min(0.75 + 0.05 * (neg_score - pos_score), 0.98)
                p_pos_raw = 0.05 + 0.02 * pos_score
                p_neg_raw = confidence
            else:
                mock_sentiment = "Neutral"
                confidence = 0.5 + 0.05 * min(pos_score, 3)
                p_pos_raw = 0.3
                p_neg_raw = 0.3
                
            # Synthesize neutral score and normalize
            p_neutral_raw = 1.0 - abs(p_pos_raw - p_neg_raw)
            total = p_pos_raw + p_neg_raw + p_neutral_raw
            
            result = {
                "sentiment": mock_sentiment,
                "confidence": round(confidence, 3),
                "probabilities": {
                    "positive": round(p_pos_raw / total, 3),
                    "negative": round(p_neg_raw / total, 3),
                    "neutral": round(p_neutral_raw / total, 3)
                },
                "inference_time_ms": 1
            }
        else:
            result = inference.predict(req.text[:2000]) # Cap at 2000 chars
            
        result['text'] = req.text
        
        # Update metrics
        total_analyses += 1
        label = result["sentiment"]
        if label == "Positive":
            positive_count += 1
        elif label == "Negative":
            negative_count += 1
        else:
            neutral_count += 1
            
        result['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        analysis_history.append(result)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": inference is not None}

@app.get("/metrics")
def metrics():
    return {
        "total_analyses": total_analyses,
        "positive": positive_count,
        "negative": negative_count,
        "neutral": neutral_count
    }

@app.post("/clear-history")
def clear_history():
    global analysis_history
    analysis_history.clear()
    return {"status": "cleared"}
