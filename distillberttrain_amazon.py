import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from DistillBert import DistillBertClassification
from torch import cuda
import argparse

import torch.nn.functional as F  

def train_model(model, train_dataloader, dev_dataloader, epochs, learning_rate):
    device = "cuda" if cuda.is_available() else "cpu"
    model.to(device)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    best_model = None
    best_accuracy = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            input_ids, attention_mask, labels = batch
            
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask).squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch: {epoch + 1}, Average Training Loss: {total_loss / len(train_dataloader)}")
        
        # Validation step
        model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for batch in dev_dataloader:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                outputs = model(input_ids, attention_mask).squeeze()
                predictions = (outputs > 0).int()
                predictions = predictions.to(labels.device)  
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
            accuracy = correct / total
            print(f"Epoch: {epoch + 1}, Validation Accuracy: {accuracy:.4f}")
            
            # Check for best accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model.state_dict()

    # Load the best model
    model.load_state_dict(best_model)
    return model, best_accuracy

def evaluate_model(model, test_dataloader):
    device = "cuda" if cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    correct, total = 0, 0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids, attention_mask).squeeze()
            predictions = (outputs > 0).int()
            predictions = predictions.to(labels.device)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
    
    accuracy = correct / total
    return accuracy, all_labels, all_predictions
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--learning_rates", nargs='+', type=float, default=[0.01, 0.001, 0.05, 0.5, 0.1])
    parser.add_argument("--batch_sizes", nargs='+', type=int, default=[16, 32])
    args = parser.parse_args()
    
    amazon_data = pd.read_csv('amazon.csv')

    # Ensure 'score' is numeric and handle errors if present
    amazon_data['rating'] = pd.to_numeric(amazon_data['rating'], errors='coerce')

    # Convert scores to binary ratings (1 if score >= 3, else 0)
    amazon_data['binary_rating'] = (amazon_data['rating'] >= 3).astype(int)

    movie = pd.read_csv("IMDB Dataset.csv")
    movie['binary_rating'] = movie['sentiment'].map({'positive': 1, 'negative': 0})
    newdf_0 = movie[movie['binary_rating'] == 0]
    newdf_1 = movie[movie['binary_rating'] == 1]

    newdf_1_downsampled = newdf_1.sample(n=500, random_state=42)
    newdf_0_downsampled = newdf_0.sample(n=500, random_state=42)
    movie = pd.concat([newdf_0_downsampled, newdf_1_downsampled])
    movie = movie.sample(frac=1, random_state=42).reset_index(drop=True)

    restaurant = pd.read_csv("Restaurant_Reviews.csv")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    def tokenize(data, batch_size):
        encodings = tokenizer(data['review_content'].fillna("").tolist(), padding=True, truncation=True, return_tensors="pt")
        labels = torch.tensor(data['binary_rating'].tolist())
        dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def movietokenize(data, batch_size):
        encodings = tokenizer(data['review'].fillna("").tolist(), padding=True, truncation=True, return_tensors="pt")
        labels = torch.tensor(data['binary_rating'].tolist())
        dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def resttokenize(data, batch_size):
        encodings = tokenizer(data['Review'].fillna("").tolist(), padding=True, truncation=True, return_tensors="pt")
        labels = torch.tensor(data['Liked'].tolist())
        dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    train_data, test_data = train_test_split(amazon_data, test_size=0.2, random_state=42)
    train_data, dev_data = train_test_split(train_data, test_size=0.4, random_state=42)
    resttrain_data, resttest_data = train_test_split(restaurant, test_size=0.2, random_state=42)
    movietrain_data, movietest_data = train_test_split(movie, test_size=0.2, random_state=42)
    
    best_dev_accuracy = 0
    best_model = None
    best_batch_size = None
    best_learning_rate = None

    for batch_size in args.batch_sizes:
        for learning_rate in args.learning_rates:
            train_dataloader = tokenize(train_data, batch_size)
            dev_dataloader = tokenize(dev_data, batch_size)
            model = DistillBertClassification(freeze_bert=True)
            
            print(f"Training with batch_size={batch_size}, learning_rate={learning_rate}")
            trained_model, dev_accuracy = train_model(model, train_dataloader, dev_dataloader, args.epochs, learning_rate)
            
            if dev_accuracy > best_dev_accuracy:
                best_dev_accuracy = dev_accuracy
                best_model = trained_model
                best_batch_size = batch_size
                best_learning_rate = learning_rate

    print(f"Best model found with batch_size={best_batch_size}, learning_rate={best_learning_rate}, dev accuracy={best_dev_accuracy:.4f}")

    # Test the best model on the test set
    test_dataloader = tokenize(test_data, best_batch_size)
    test_accuracy, test_labels, test_predictions = evaluate_model(best_model, test_dataloader)
    print(f"Model accuracy on test data: {test_accuracy:.4f}")

    resttest_dataloader = resttokenize(resttest_data, best_batch_size)
    resttest_accuracy, resttest_labels, resttest_predictions = evaluate_model(best_model, resttest_dataloader)
    print(f"Model accuracy on Restaurant test data: {resttest_accuracy:.4f}")

    movietest_dataloader = movietokenize(movietest_data, best_batch_size)
    movietest_accuracy, movietest_labels, movietest_predictions = evaluate_model(best_model, movietest_dataloader)
    print(f"Model accuracy on Movie test data: {movietest_accuracy:.4f}")

if __name__ == "__main__":
    main()