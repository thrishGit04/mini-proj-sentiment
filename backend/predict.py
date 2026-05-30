"""
Standalone prediction script for Sentiment Analysis Tool.
Usage:
    python predict.py                    # Interactive mode
    python predict.py "Your text here"   # Direct prediction
"""

import sys
import os
import torch
import argparse
from transformers import DistilBertTokenizer
from model import CustomDistilBertForClassification


class SentimentPredictor:
    def __init__(self, model_dir="saved_model"):
        """Initialize the sentiment predictor with a trained model."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = model_dir
        
        # Check if model exists
        if not os.path.exists(os.path.join(model_dir, "sentiment_model.pth")):
            raise FileNotFoundError(
                f"❌ Model not found in {model_dir}. Please run 'python train.py' first."
            )
        
        print(f"📦 Loading model from {model_dir}...")
        
        # Load tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            os.path.join(model_dir, "tokenizer")
        )
        self.model = CustomDistilBertForClassification(num_labels=2, dropout_prob=0.3)
        self.model.load_state_dict(
            torch.load(
                os.path.join(model_dir, "sentiment_model.pth"),
                map_location=self.device,
                weights_only=False
            )
        )
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded successfully on {self.device}")
    
    def predict(self, text):
        """Predict sentiment of given text."""
        if not text.strip():
            print("❌ Error: Empty text provided.")
            return None
        
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
        
        # Synthesize neutral probability
        p_neutral_raw = 1.0 - abs(p_pos_raw - p_neg_raw)
        
        # Normalize
        total = p_pos_raw + p_neg_raw + p_neutral_raw
        p_pos = p_pos_raw / total
        p_neg = p_neg_raw / total
        p_neutral = p_neutral_raw / total
        
        # Determine sentiment label
        highest_prob = max(p_pos, p_neg, p_neutral)
        if highest_prob == p_pos:
            sentiment = "Positive"
            emoji = "😊"
        elif highest_prob == p_neg:
            sentiment = "Negative"
            emoji = "😞"
        else:
            sentiment = "Neutral"
            emoji = "😐"
        
        return {
            "text": text,
            "sentiment": sentiment,
            "emoji": emoji,
            "confidence": round(highest_prob * 100, 2),
            "probabilities": {
                "positive": round(p_pos * 100, 2),
                "negative": round(p_neg * 100, 2),
                "neutral": round(p_neutral * 100, 2)
            }
        }
    
    def display_result(self, result):
        """Display prediction result in a formatted way."""
        if result is None:
            return
        
        print("\n" + "=" * 70)
        print(f"📝 Text: {result['text'][:100]}{'...' if len(result['text']) > 100 else ''}")
        print("=" * 70)
        print(f"\n🎯 Sentiment: {result['emoji']} {result['sentiment']}")
        print(f"📊 Confidence: {result['confidence']}%")
        print(f"\n📈 Probability Breakdown:")
        print(f"   • Positive: {result['probabilities']['positive']}%")
        print(f"   • Negative: {result['probabilities']['negative']}%")
        print(f"   • Neutral:  {result['probabilities']['neutral']}%")
        print("\n" + "=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Predict sentiment of text using trained DistilBERT model"
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to analyze (optional; if not provided, enters interactive mode)"
    )
    parser.add_argument(
        "--model-dir",
        default="saved_model",
        help="Path to saved model directory (default: saved_model)"
    )
    
    args = parser.parse_args()
    
    try:
        predictor = SentimentPredictor(model_dir=args.model_dir)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    if args.text:
        # Direct prediction mode
        result = predictor.predict(args.text)
        predictor.display_result(result)
    else:
        # Interactive mode
        print("\n🎯 Sentiment Analysis Tool - Interactive Mode")
        print("Type 'quit' or 'exit' to stop.\n")
        
        while True:
            try:
                text = input("Enter text to analyze: ").strip()
                if text.lower() in ['quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                
                result = predictor.predict(text)
                predictor.display_result(result)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
