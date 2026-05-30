import torch
import time
from transformers import DistilBertTokenizer
from model import CustomDistilBertForClassification
import os

class SentimentInference:
    def __init__(self, model_dir="saved_model"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not os.path.exists(os.path.join(model_dir, "sentiment_model.pth")):
            raise FileNotFoundError(f"Model not found in {model_dir}. Please run train.py first.")
            
        self.tokenizer = DistilBertTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
        self.model = CustomDistilBertForClassification(num_labels=2, dropout_prob=0.3)
        self.model.load_state_dict(torch.load(os.path.join(model_dir, "sentiment_model.pth"), map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, text):
        start_time = time.time()
        
        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=256,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
        p_neg_raw = float(probs[0])
        p_pos_raw = float(probs[1])
        
        # Synthesize a neutral probability based on the proximity of pos and neg
        p_neutral_raw = 1.0 - abs(p_pos_raw - p_neg_raw)
        
        # Normalize to sum to 1
        total = p_pos_raw + p_neg_raw + p_neutral_raw
        p_pos = p_pos_raw / total
        p_neg = p_neg_raw / total
        p_neutral = p_neutral_raw / total
        
        # Determine final label
        highest_prob = max(p_pos, p_neg, p_neutral)
        if highest_prob == p_pos:
            label = "Positive"
        elif highest_prob == p_neg:
            label = "Negative"
        else:
            label = "Neutral"
            
        inference_time_ms = int((time.time() - start_time) * 1000)
        
        return {
            "sentiment": label,
            "confidence": round(highest_prob, 3),
            "probabilities": {
                "positive": round(p_pos, 3),
                "negative": round(p_neg, 3),
                "neutral": round(p_neutral, 3)
            },
            "inference_time_ms": inference_time_ms
        }
