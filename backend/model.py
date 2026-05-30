import torch
import torch.nn as nn
from transformers import DistilBertModel

class CustomDistilBertForClassification(nn.Module):
    def __init__(self, num_labels=2, dropout_prob=0.3):
        super(CustomDistilBertForClassification, self).__init__()
        self.num_labels = num_labels
        
        # Load pre-trained model
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        
        # Freeze first 4 layers
        for layer in self.distilbert.transformer.layer[:4]:
            for param in layer.parameters():
                param.requires_grad = False
                
        # Classification head
        self.pre_classifier = nn.Linear(self.distilbert.config.hidden_size, self.distilbert.config.hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_labels)
        self.relu = nn.ReLU()
        
    def forward(self, input_ids, attention_mask=None):
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract the hidden state of the [CLS] token (index 0)
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        
        # Pass through pre-classifier
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = self.relu(pooled_output)
        
        # Pass through dropout and classifier
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
