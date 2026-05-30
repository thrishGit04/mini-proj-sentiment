from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import datetime
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# In-memory history for session logging
analysis_history = []

# Load the inference model
inference = None
try:
    from inference import SentimentInference
    model_path = os.path.join(os.path.dirname(__file__), "saved_model")
    inference = SentimentInference(model_dir=model_path)
except Exception as e:
    logger.error(f"❌ Failed to initialize SentimentInference: {e}")

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    global total_analyses, positive_count, negative_count, neutral_count
    
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="No text provided")
        
    try:
        if not inference:
            raise HTTPException(status_code=500, detail="Sentiment analysis engine is not available")
            
        # Run prediction (internally uses fine-tuned model or keyword fallback)
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
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": inference is not None and inference.is_loaded}

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
