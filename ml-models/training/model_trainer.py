import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, roc_auc_score
from typing import Dict, Tuple
import json

class ModelTrainer:
    """Train and validate ML models with K-fold cross-validation"""
    
    def __init__(self, device='cpu'):
        self.device = device
        
    def train_voice_model(self, model, X: np.ndarray, y: np.ndarray, epochs: int = 50) -> Dict:
        """Train voice model with cross-validation"""
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        f1_scores, auc_scores = [], []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)
            y_train_tensor = torch.LongTensor(y_train)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_val).unsqueeze(1)
                val_outputs = model(X_val_tensor)
                val_preds = torch.argmax(val_outputs, dim=1).numpy()
                
            f1 = f1_score(y_val, val_preds, average='weighted')
            auc = roc_auc_score(y_val, torch.softmax(val_outputs, dim=1).numpy(), multi_class='ovr')
            
            f1_scores.append(f1)
            auc_scores.append(auc)
        
        return {
            'f1_mean': np.mean(f1_scores),
            'f1_std': np.std(f1_scores),
            'auc_mean': np.mean(auc_scores),
            'auc_std': np.std(auc_scores)
        }
    
    def train_text_model(self, model, tokenizer, texts: list, labels: np.ndarray, epochs: int = 3) -> Dict:
        """Train BERT model with cross-validation"""
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        f1_scores, auc_scores = [], []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(texts)):
            train_texts = [texts[i] for i in train_idx]
            val_texts = [texts[i] for i in val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            # Tokenize
            train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')
            val_encodings = tokenizer(val_texts, truncation=True, padding=True, return_tensors='pt')
            
            optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
            criterion = nn.CrossEntropyLoss()
            
            # Training
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(train_encodings['input_ids'], train_encodings['attention_mask'])
                loss = criterion(outputs, torch.LongTensor(y_train))
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(val_encodings['input_ids'], val_encodings['attention_mask'])
                val_preds = torch.argmax(val_outputs, dim=1).numpy()
            
            f1 = f1_score(y_val, val_preds, average='weighted')
            auc = roc_auc_score(y_val, torch.softmax(val_outputs, dim=1).numpy(), multi_class='ovr')
            
            f1_scores.append(f1)
            auc_scores.append(auc)
        
        return {
            'f1_mean': np.mean(f1_scores),
            'auc_mean': np.mean(auc_scores)
        }
    
    def save_model(self, model, path: str, metadata: Dict = None):
        """Save trained model"""
        torch.save({
            'model_state_dict': model.state_dict(),
            'metadata': metadata or {}
        }, path)