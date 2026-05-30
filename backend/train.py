import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch.nn as nn
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, classification_report
from model import CustomDistilBertForClassification
import argparse

class IMDbDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_model(epochs=3, batch_size=32):
    print("Downloading IMDb Dataset...")
    path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    csv_path = os.path.join(path, "IMDB Dataset.csv")
    
    print("Loading and preparing data...")
    df = pd.read_csv(csv_path)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    # Take 5k samples total (2.5k pos, 2.5k neg) for training, and 1k for testing to keep it fast
    pos_df = df[df['sentiment'] == 1].sample(2500, random_state=42)
    neg_df = df[df['sentiment'] == 0].sample(2500, random_state=42)
    
    # Additional 1k for testing (500 pos, 500 neg)
    remaining_pos = df[(df['sentiment'] == 1) & (~df.index.isin(pos_df.index))].sample(500, random_state=42)
    remaining_neg = df[(df['sentiment'] == 0) & (~df.index.isin(neg_df.index))].sample(500, random_state=42)
    
    train_df = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = pd.concat([remaining_pos, remaining_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Training on {len(train_df)} samples, Evaluating on {len(test_df)} samples.")
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    train_dataset = IMDbDataset(train_df.review.to_numpy(), train_df.sentiment.to_numpy(), tokenizer)
    test_dataset = IMDbDataset(test_df.review.to_numpy(), test_df.sentiment.to_numpy(), tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = CustomDistilBertForClassification(num_labels=2, dropout_prob=0.3)
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    best_accuracy = 0
    patience = 1
    patience_counter = 0
    
    print("Starting training...")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        total_train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            model.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            total_train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")
                
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Evaluation
        model.eval()
        val_preds = []
        val_labels = []
        val_probs = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
                
        acc = accuracy_score(val_labels, val_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='binary')
        auc = roc_auc_score(val_labels, val_probs)
        
        print(f"  Avg Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        
        if acc > best_accuracy:
            best_accuracy = acc
            patience_counter = 0
            
            # Save Model
            os.makedirs("saved_model", exist_ok=True)
            torch.save(model.state_dict(), "saved_model/sentiment_model.pth")
            tokenizer.save_pretrained("saved_model/tokenizer")
            
            metrics = {
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": auc,
                "confusion_matrix": confusion_matrix(val_labels, val_preds).tolist(),
                "classification_report": classification_report(val_labels, val_preds, output_dict=True)
            }
            with open("saved_model/metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)
            print("  Saved best model!")
        else:
            patience_counter += 1
            if patience_counter > patience:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    train_model(epochs=args.epochs, batch_size=args.batch_size)
