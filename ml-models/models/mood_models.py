import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, roc_auc_score
from typing import Tuple, Dict

class VoiceMoodLSTM(nn.Module):
    """LSTM model for voice-based mood detection"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, num_classes: int = 5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(self.dropout(lstm_out[:, -1, :]))
        return output

class VoiceMoodCNN(nn.Module):
    """CNN model for voice-based mood detection"""
    
    def __init__(self, input_channels: int = 1, num_classes: int = 5):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(128 * 32, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(self.dropout(x)))
        x = self.fc2(self.dropout(x))
        return x

class BERTMoodClassifier(nn.Module):
    """Fine-tuned BERT for depression/loneliness detection"""
    
    def __init__(self, bert_model_name: str = 'bert-base-uncased', num_classes: int = 5):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.classifier(self.dropout(pooled_output))
        return output

class CrisisInterventionScorer:
    """SVM/Random Forest for crisis intervention scoring"""
    
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.svm_model = SVC(probability=True, random_state=42)
        
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Train both models and return metrics"""
        # Train Random Forest
        rf_scores = cross_val_score(self.rf_model, X, y, cv=5, scoring='f1_weighted')
        self.rf_model.fit(X, y)
        
        # Train SVM
        svm_scores = cross_val_score(self.svm_model, X, y, cv=5, scoring='f1_weighted')
        self.svm_model.fit(X, y)
        
        return {
            'rf_f1': rf_scores.mean(),
            'svm_f1': svm_scores.mean()
        }
    
    def predict_crisis_risk(self, features: np.ndarray) -> Dict[str, float]:
        """Predict crisis risk using ensemble"""
        rf_prob = self.rf_model.predict_proba(features)[0]
        svm_prob = self.svm_model.predict_proba(features)[0]
        
        # Ensemble prediction
        ensemble_prob = (rf_prob + svm_prob) / 2
        crisis_risk = ensemble_prob[-1] if len(ensemble_prob) > 1 else ensemble_prob[0]
        
        return {
            'crisis_risk': float(crisis_risk),
            'risk_level': 'high' if crisis_risk > 0.7 else 'moderate' if crisis_risk > 0.4 else 'low'
        }

class ModelEnsemble:
    """Ensemble methods for improved accuracy"""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        
    def add_model(self, name: str, model, weight: float = 1.0):
        """Add model to ensemble"""
        self.models[name] = model
        self.weights[name] = weight
        
    def predict(self, voice_features: np.ndarray = None, text_features: np.ndarray = None) -> Dict[str, float]:
        """Ensemble prediction from multiple models"""
        predictions = {}
        total_weight = 0
        
        if voice_features is not None and 'voice_lstm' in self.models:
            pred = self.models['voice_lstm'](torch.FloatTensor(voice_features))
            predictions['voice'] = torch.softmax(pred, dim=1).detach().numpy()[0]
            total_weight += self.weights['voice_lstm']
            
        if text_features is not None and 'bert' in self.models:
            pred = self.models['bert'](text_features)
            predictions['text'] = torch.softmax(pred, dim=1).detach().numpy()[0]
            total_weight += self.weights['bert']
        
        # Weighted average
        if predictions:
            ensemble_pred = np.zeros(5)  # 5 mood classes
            for model_name, pred in predictions.items():
                weight = self.weights.get(model_name.split('_')[0] + '_lstm' if 'voice' in model_name else 'bert', 1.0)
                ensemble_pred += pred * weight
            
            ensemble_pred /= total_weight
            mood_class = np.argmax(ensemble_pred)
            confidence = float(ensemble_pred[mood_class])
            
            return {
                'mood_class': int(mood_class),
                'confidence': confidence,
                'probabilities': ensemble_pred.tolist()
            }
        
        return {'mood_class': 2, 'confidence': 0.5, 'probabilities': [0.2] * 5}