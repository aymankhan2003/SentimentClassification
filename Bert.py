import torch
import torch.nn as nn
from transformers import AutoModel

class BertSentimentModel(nn.Module):
    """Model for classifying restaurant reviews as good or bad based on BERT embeddings."""
    def __init__(self, freeze_bert: bool = False):
        super().__init__()
        
        # Load a pre-trained DistilBERT model
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")

        # Get the number of features in the BERT output
        bert_output_dim = self.bert.config.hidden_size
        #Set a random seed
        # Define the sequence of linear layers
        self.linear_layers = nn.ModuleList([
            nn.Linear(bert_output_dim, 100),  # First linear layer
            nn.Linear(100, 1)  # Output layer for binary classification
        ])

        # Option to freeze BERT parameters to prevent them from being updated during training
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # Pass input through BERT
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the [CLS] token's embedding
        cls_embedding = bert_output.last_hidden_state[:, 0, :]

        # Pass the [CLS] token embedding through the linear layers with a sigmoid activation function
        x = cls_embedding
        for layer in self.linear_layers[:-1]:
            x = torch.relu(layer(x))  # Using ReLU for hidden layers
        x = torch.sigmoid(self.linear_layers[-1](x)) # Sigmoid activation for the output layer
            # Softmax over the correct dimension

        return x.squeeze()  # Remove any extra dimensions
