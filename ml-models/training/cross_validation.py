import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from typing import Dict, List, Tuple, Any, Callable
import torch
from dataclasses import dataclass

@dataclass
class CVResults:
    """Cross-validation results container"""
    f1_scores: List[float]
    accuracy_scores: List[float]
    precision_scores: List[float]
    recall_scores: List[float]
    auc_scores: List[float]
    
    @property
    def mean_f1(self) -> float:
        return np.mean(self.f1_scores)
    
    @property
    def std_f1(self) -> float:
        return np.std(self.f1_scores)
    
    @property
    def mean_accuracy(self) -> float:
        return np.mean(self.accuracy_scores)
    
    @property
    def std_accuracy(self) -> float:
        return np.std(self.accuracy_scores)
    
    @property
    def mean_auc(self) -> float:
        return np.mean(self.auc_scores)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'f1_mean': self.mean_f1,
            'f1_std': self.std_f1,
            'accuracy_mean': self.mean_accuracy,
            'accuracy_std': self.std_accuracy,
            'auc_mean': self.mean_auc,
            'precision_mean': np.mean(self.precision_scores),
            'recall_mean': np.mean(self.recall_scores),
            'all_scores': {
                'f1': self.f1_scores,
                'accuracy': self.accuracy_scores,
                'precision': self.precision_scores,
                'recall': self.recall_scores,
                'auc': self.auc_scores
            }
        }

class CrossValidator:
    """Advanced cross-validation for mood analysis models"""
    
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    def validate_pytorch_model(self, model_class: type, features: np.ndarray, 
                             labels: np.ndarray, train_func: Callable,
                             predict_func: Callable, **model_kwargs) -> CVResults:
        """Cross-validate PyTorch models"""
        f1_scores = []
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        auc_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(self.skf.split(features, labels)):
            print(f"Fold {fold + 1}/{self.n_splits}")
            
            X_train, X_val = features[train_idx], features[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            # Create and train model
            model = model_class(**model_kwargs)
            trained_model = train_func(model, X_train, y_train)
            
            # Make predictions
            predictions, probabilities = predict_func(trained_model, X_val)
            
            # Calculate metrics
            f1 = f1_score(y_val, predictions, average='weighted')
            accuracy = accuracy_score(y_val, predictions)
            precision = precision_score(y_val, predictions, average='weighted', zero_division=0)
            recall = recall_score(y_val, predictions, average='weighted', zero_division=0)
            
            # AUC for multiclass
            try:
                auc = roc_auc_score(y_val, probabilities, multi_class='ovr', average='weighted')
            except:
                auc = 0.0
            
            f1_scores.append(f1)
            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            auc_scores.append(auc)
            
            print(f"  F1: {f1:.4f}, Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
        
        return CVResults(f1_scores, accuracy_scores, precision_scores, recall_scores, auc_scores)
    
    def validate_sklearn_model(self, model, features: np.ndarray, labels: np.ndarray) -> CVResults:
        """Cross-validate sklearn models"""
        f1_scores = cross_val_score(model, features, labels, cv=self.skf, scoring='f1_weighted')
        accuracy_scores = cross_val_score(model, features, labels, cv=self.skf, scoring='accuracy')
        precision_scores = cross_val_score(model, features, labels, cv=self.skf, scoring='precision_weighted')
        recall_scores = cross_val_score(model, features, labels, cv=self.skf, scoring='recall_weighted')
        
        # Calculate AUC manually for multiclass
        auc_scores = []
        for train_idx, val_idx in self.skf.split(features, labels):
            X_train, X_val = features[train_idx], features[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            model.fit(X_train, y_train)
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_val)
                try:
                    auc = roc_auc_score(y_val, probabilities, multi_class='ovr', average='weighted')
                except:
                    auc = 0.0
            else:
                auc = 0.0
            auc_scores.append(auc)
        
        return CVResults(
            f1_scores.tolist(), 
            accuracy_scores.tolist(),
            precision_scores.tolist(),
            recall_scores.tolist(),
            auc_scores
        )
    
    def validate_voice_lstm(self, model_class, features: np.ndarray, labels: np.ndarray,
                           input_size: int, device: str = 'cpu') -> CVResults:
        """Specialized validation for voice LSTM models"""
        
        def train_func(model, X_train, y_train):
            model = model.to(device)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Create data loader
            dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X_train),
                torch.LongTensor(y_train)
            )
            loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
            
            model.train()
            for epoch in range(20):  # Reduced epochs for CV
                for batch_features, batch_labels in loader:
                    batch_features = batch_features.to(device).unsqueeze(1)  # Add sequence dimension
                    batch_labels = batch_labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()
            
            return model
        
        def predict_func(model, X_val):
            model.eval()
            predictions = []
            probabilities = []
            
            with torch.no_grad():
                for i in range(len(X_val)):
                    feature_tensor = torch.FloatTensor(X_val[i]).unsqueeze(0).unsqueeze(1).to(device)
                    output = model(feature_tensor)
                    
                    prob = torch.softmax(output, dim=1).cpu().numpy()[0]
                    pred = torch.argmax(output, dim=1).cpu().numpy()[0]
                    
                    predictions.append(pred)
                    probabilities.append(prob)
            
            return np.array(predictions), np.array(probabilities)
        
        return self.validate_pytorch_model(
            model_class, features, labels, train_func, predict_func,
            input_size=input_size
        )
    
    def validate_voice_cnn(self, model_class, features: np.ndarray, labels: np.ndarray,
                          input_size: int, device: str = 'cpu') -> CVResults:
        """Specialized validation for voice CNN models"""
        
        def train_func(model, X_train, y_train):
            model = model.to(device)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X_train),
                torch.LongTensor(y_train)
            )
            loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
            
            model.train()
            for epoch in range(20):
                for batch_features, batch_labels in loader:
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()
            
            return model
        
        def predict_func(model, X_val):
            model.eval()
            predictions = []
            probabilities = []
            
            with torch.no_grad():
                for i in range(len(X_val)):
                    feature_tensor = torch.FloatTensor(X_val[i]).unsqueeze(0).to(device)
                    output = model(feature_tensor)
                    
                    prob = torch.softmax(output, dim=1).cpu().numpy()[0]
                    pred = torch.argmax(output, dim=1).cpu().numpy()[0]
                    
                    predictions.append(pred)
                    probabilities.append(prob)
            
            return np.array(predictions), np.array(probabilities)
        
        return self.validate_pytorch_model(
            model_class, features, labels, train_func, predict_func,
            input_size=input_size
        )
    
    def validate_bert_model(self, model_class, texts: List[str], labels: List[int],
                           tokenizer, device: str = 'cpu', max_length: int = 512) -> CVResults:
        """Specialized validation for BERT models"""
        
        def train_func(model, train_texts, train_labels):
            from transformers import Trainer, TrainingArguments
            
            model = model.to(device)
            
            # Prepare dataset
            encodings = tokenizer(
                train_texts, truncation=True, padding=True, 
                max_length=max_length, return_tensors='pt'
            )
            
            class Dataset(torch.utils.data.Dataset):
                def __init__(self, encodings, labels):
                    self.encodings = encodings
                    self.labels = labels
                
                def __getitem__(self, idx):
                    item = {key: val[idx] for key, val in self.encodings.items()}
                    item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
                    return item
                
                def __len__(self):
                    return len(self.labels)
            
            dataset = Dataset(encodings, train_labels)
            
            training_args = TrainingArguments(
                output_dir='./temp_bert',
                num_train_epochs=2,  # Reduced for CV
                per_device_train_batch_size=8,
                warmup_steps=100,
                weight_decay=0.01,
                logging_steps=50,
                save_steps=1000,
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
            )
            
            trainer.train()
            return model
        
        def predict_func(model, val_texts):
            model.eval()
            predictions = []
            probabilities = []
            
            with torch.no_grad():
                for text in val_texts:
                    encoding = tokenizer(
                        text, return_tensors='pt', truncation=True, 
                        padding=True, max_length=max_length
                    )
                    
                    input_ids = encoding['input_ids'].to(device)
                    attention_mask = encoding['attention_mask'].to(device)
                    
                    output = model(input_ids, attention_mask)
                    
                    prob = torch.softmax(output, dim=1).cpu().numpy()[0]
                    pred = torch.argmax(output, dim=1).cpu().numpy()[0]
                    
                    predictions.append(pred)
                    probabilities.append(prob)
            
            return np.array(predictions), np.array(probabilities)
        
        # Convert texts to indices for stratification
        text_indices = list(range(len(texts)))
        
        f1_scores = []
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        auc_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(self.skf.split(text_indices, labels)):
            print(f"BERT Fold {fold + 1}/{self.n_splits}")
            
            train_texts = [texts[i] for i in train_idx]
            val_texts = [texts[i] for i in val_idx]
            train_labels = [labels[i] for i in train_idx]
            val_labels = [labels[i] for i in val_idx]
            
            # Create and train model
            model = model_class()
            trained_model = train_func(model, train_texts, train_labels)
            
            # Make predictions
            predictions, probabilities = predict_func(trained_model, val_texts)
            
            # Calculate metrics
            f1 = f1_score(val_labels, predictions, average='weighted')
            accuracy = accuracy_score(val_labels, predictions)
            precision = precision_score(val_labels, predictions, average='weighted', zero_division=0)
            recall = recall_score(val_labels, predictions, average='weighted', zero_division=0)
            
            try:
                auc = roc_auc_score(val_labels, probabilities, multi_class='ovr', average='weighted')
            except:
                auc = 0.0
            
            f1_scores.append(f1)
            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            auc_scores.append(auc)
            
            print(f"  F1: {f1:.4f}, Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
        
        return CVResults(f1_scores, accuracy_scores, precision_scores, recall_scores, auc_scores)
    
    def compare_models(self, results: Dict[str, CVResults]) -> Dict[str, Any]:
        """Compare multiple model results"""
        comparison = {}
        
        for name, result in results.items():
            comparison[name] = {
                'f1_mean': result.mean_f1,
                'f1_std': result.std_f1,
                'accuracy_mean': result.mean_accuracy,
                'accuracy_std': result.std_accuracy,
                'auc_mean': result.mean_auc
            }
        
        # Find best models
        best_f1_model = max(results.keys(), key=lambda k: results[k].mean_f1)
        best_accuracy_model = max(results.keys(), key=lambda k: results[k].mean_accuracy)
        
        comparison['summary'] = {
            'best_f1_model': best_f1_model,
            'best_f1_score': results[best_f1_model].mean_f1,
            'best_accuracy_model': best_accuracy_model,
            'best_accuracy_score': results[best_accuracy_model].mean_accuracy
        }
        
        return comparison