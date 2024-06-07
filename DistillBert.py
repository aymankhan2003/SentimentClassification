import torch
import torch.nn as nn
from transformers import AutoModel

class DistillBertClassification(nn.Module):
    """A classifier that uses DistilBERT to classify restaurant reviews into good or bad."""
    def __init__(self, freeze_bert=True):
        super().__init__()
        # Load DistilBERT with pretrained weights
        self.distilbert = AutoModel.from_pretrained("distilbert-base-uncased")

        # Get the number of hidden units in DistilBERT output
        hidden_size = self.distilbert.config.hidden_size

        # Output layer that maps from hidden size to 1 (binary classification)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

        # Option to freeze DistilBERT parameters during training
        if freeze_bert:
            for param in self.distilbert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # Forward pass through DistilBERT
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get the last hidden state of the [CLS] token for classification
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Pass the [CLS] token representation through the classifier
        logits = self.classifier(cls_output)
        # Output the logits for binary classification
        return logits.squeeze()  # Squeeze to remove any extra dimensions
