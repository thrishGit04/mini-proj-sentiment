import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer
import os
import logging
import re
import time
import math
from model import CustomDistilBertForClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentInference:
    def __init__(self, model_dir="saved_model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded = False
        self.use_fallback = True
        self.model_dir = model_dir
        
        # Try to load model, but don't crash if it fails
        try:
            model_path = os.path.join(model_dir, "sentiment_model.pth")
            tokenizer_path = os.path.join(model_dir, "tokenizer")
            
            if os.path.exists(model_path):
                # Try local tokenizer first, fallback to online uncased DistilBERT
                if os.path.exists(tokenizer_path):
                    self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
                else:
                    self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
                
                self.model = CustomDistilBertForClassification(num_labels=2, dropout_prob=0.3)
                self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
                self.model.to(self.device)
                self.model.eval()
                self.is_loaded = True
                self.use_fallback = False
                logger.info(f"✅ Neural model loaded successfully from {model_path}")
            else:
                logger.warning(f"⚠️ Model file not found at {model_path}, using keyword fallback")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load model: {e}, using fallback")
        
        # Expanded keyword lists for better detection
        self.positive_words = ['love', 'great', 'amazing', 'excellent', 'good', 'fantastic', 
                               'awesome', 'perfect', 'wonderful', 'best', 'fabulous', 'brilliant',
                               'beautiful', 'nice', 'enjoyable', 'outstanding', 'superb', 'incredible',
                               'delightful', 'masterpiece', 'recommend', 'glad', 'happy', 'impressed']
        
        self.negative_words = ['hate', 'bad', 'terrible', 'awful', 'worst', 'poor', 
                               'horrible', 'disappointing', 'boring', 'waste', 'worse',
                               'waste of time', 'avoid', 'annoying', 'frustrating', 'mediocre',
                               'stupid', 'ridiculous', 'dislike', 'unfortunately', 'sad']
        
        # Contradiction indicators (words and punctuation patterns)
        self.contradiction_words = [' but ', ' however ', ' although ', ' though ', ' yet ', ' except ']
        self.punctuation_splitters = ['.', '!', '?', ';', ',', '.\n', '!\n', '?\n']
    
    def predict(self, text, max_length=256):
        start_time = time.time()
        
        # Cap text at 2000 characters
        text = text[:2000]
        
        # FIRST: Check for mixed sentiment using advanced detection
        mixed_result = self._detect_mixed_sentiment_advanced(text)
        
        if mixed_result["is_mixed"]:
            # Use the averaged scores from mixed detection
            prob_positive_raw = mixed_result["avg_positive"]
            prob_negative_raw = mixed_result["avg_negative"]
            
            # For mixed sentiment, strongly boost neutral
            prob_neutral_raw = 1.0 - abs(prob_positive_raw - prob_negative_raw)
            total = prob_positive_raw + prob_negative_raw + prob_neutral_raw
            
            prob_positive_norm = prob_positive_raw / total
            prob_negative_norm = prob_negative_raw / total
            prob_neutral_norm = prob_neutral_raw / total
            
            # Boost neutral further for clear contradictions
            if mixed_result.get("method") == "clause_split":
                prob_neutral_norm = prob_neutral_norm * 1.3
                # Renormalize
                total_norm = prob_positive_norm + prob_negative_norm + prob_neutral_norm
                prob_positive_norm = prob_positive_norm / total_norm
                prob_negative_norm = prob_negative_norm / total_norm
                prob_neutral_norm = prob_neutral_norm / total_norm
            
            # Determine label
            sentiments = ["Negative", "Positive", "Neutral"]
            values = [prob_negative_norm, prob_positive_norm, prob_neutral_norm]
            label = sentiments[values.index(max(values))]
            
            inference_time_ms = int((time.time() - start_time) * 1000)
            
            return {
                "sentiment": label,
                "confidence": round(max(values), 3),
                "probabilities": {
                    "positive": round(prob_positive_norm, 3),
                    "negative": round(prob_negative_norm, 3),
                    "neutral": round(prob_neutral_norm, 3)
                },
                "inference_time_ms": max(1, inference_time_ms)
            }
        
        # SECOND: Standard prediction for non-mixed text
        prob_positive, prob_negative = self._predict_clause_probs(text, max_length)
        result = self._process_probabilities(prob_positive, prob_negative)
        
        inference_time_ms = int((time.time() - start_time) * 1000)
        result["inference_time_ms"] = max(1, inference_time_ms)
        return result
    
    def _predict_clause_probs(self, text, max_length=256):
        """Predict raw positive and negative probabilities for a single text segment"""
        if not self.use_fallback and self.is_loaded:
            try:
                inputs = self.tokenizer(
                    text,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                
                return float(probs[1]), float(probs[0]) # positive, negative
            except Exception as e:
                logger.error(f"Clause model prediction failed: {e}, using keyword scores")
                
        # Keyword-based fallback scoring
        return self._get_keyword_probabilities(text)

    def _get_keyword_probabilities(self, text):
        text_lower = text.lower()
        positive_score = sum(1 for word in self.positive_words if word in text_lower)
        negative_score = sum(1 for word in self.negative_words if word in text_lower)
        
        diff = positive_score - negative_score
        
        # Exponential scaling to map score difference to realistic probability margins
        exp_pos = math.exp(1.5 * diff)
        exp_neg = math.exp(-1.5 * diff)
        
        prob_positive_raw = exp_pos / (exp_pos + exp_neg)
        prob_negative_raw = exp_neg / (exp_pos + exp_neg)
        return prob_positive_raw, prob_negative_raw
    
    def _split_into_clauses(self, text):
        """Split text into clauses using punctuation and conjunctions"""
        text_lower = text.lower()
        
        # First split by punctuation
        clauses = [text_lower]
        for splitter in self.punctuation_splitters:
            new_clauses = []
            for clause in clauses:
                new_clauses.extend(clause.split(splitter))
            clauses = new_clauses
        
        # Split by common conjunctions as well
        final_clauses = []
        for clause in clauses:
            if not clause.strip():
                continue
            # Split by 'but', 'however', 'although', 'though', 'yet', 'except', 'while', 'whereas'
            sub_parts = re.split(r'\s+(?:but|however|although|though|yet|except|while|whereas)\s+', clause.strip())
            final_clauses.extend([p.strip() for p in sub_parts if p.strip()])
        
        return [c for c in final_clauses if c and len(c) > 3]
    
    def _detect_mixed_sentiment_advanced(self, text):
        """Advanced mixed sentiment detection using clause analysis"""
        text_lower = text.lower()
        
        # Method 1: Check for contradiction words
        has_contradiction_word = any(word in text_lower for word in self.contradiction_words)
        
        # Method 2: Split into clauses and analyze each
        clauses = self._split_into_clauses(text)
        
        if len(clauses) >= 2:
            clause_scores = []
            for clause in clauses:
                pos, neg = self._predict_clause_probs(clause)
                clause_scores.append((pos, neg))
            
            # Check if we have both positive and negative clauses
            has_positive_clause = any(pos > neg + 0.05 for pos, neg in clause_scores)
            has_negative_clause = any(neg > pos + 0.05 for pos, neg in clause_scores)
            
            if has_positive_clause and has_negative_clause:
                # Calculate average sentiment across all clauses
                avg_pos = sum(p for p, _ in clause_scores) / len(clause_scores)
                avg_neg = sum(n for _, n in clause_scores) / len(clause_scores)
                
                return {
                    "is_mixed": True,
                    "avg_positive": avg_pos,
                    "avg_negative": avg_neg,
                    "method": "clause_split"
                }
        
        # Method 3: Check for overall positive AND negative keywords with significant counts
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        
        if positive_count >= 1 and negative_count >= 1 and has_contradiction_word:
            # Get overall scores
            prob_positive_raw, prob_negative_raw = self._predict_clause_probs(text)
            
            return {
                "is_mixed": True,
                "avg_positive": prob_positive_raw,
                "avg_negative": prob_negative_raw,
                "method": "keyword_balance"
            }
        
        return {"is_mixed": False}
    
    def _process_probabilities(self, prob_positive, prob_negative):
        # Neutral synthesis
        prob_neutral_raw = 1.0 - abs(prob_positive - prob_negative)
        total = prob_positive + prob_negative + prob_neutral_raw
        prob_positive_norm = prob_positive / total
        prob_negative_norm = prob_negative / total
        prob_neutral_norm = prob_neutral_raw / total
        
        probabilities = {
            "positive": round(prob_positive_norm, 3),
            "negative": round(prob_negative_norm, 3),
            "neutral": round(prob_neutral_norm, 3)
        }
        
        sentiments = ["Negative", "Positive", "Neutral"]
        values = [prob_negative_norm, prob_positive_norm, prob_neutral_norm]
        sentiment = sentiments[values.index(max(values))]
        
        return {
            "sentiment": sentiment,
            "confidence": round(max(values), 3),
            "probabilities": probabilities
        }
