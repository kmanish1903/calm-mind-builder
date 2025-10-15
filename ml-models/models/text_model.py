import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score
import numpy as np
from typing import Dict, List, Tuple

class BERTMoodClassifier(nn.Module):
    """Fine-tuned BERT for depression/loneliness detection"""
    
    def __init__(self, model_name: str = "mental/mental-bert-base-uncased", num_classes: int = 5):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

class CrisisInterventionScorer:
    """SVM/Random Forest for crisis intervention scoring"""
    
    def __init__(self, model_type: str = 'svm'):
        self.model_type = model_type
        if model_type == 'svm':
            self.model = SVC(kernel='rbf', probability=True, random_state=42)
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Crisis risk levels: 0=low, 1=moderate, 2=high, 3=critical
        self.risk_levels = ['low', 'moderate', 'high', 'critical']
        
    def extract_crisis_features(self, texts: List[str]) -> np.ndarray:
        """Extract features specifically for crisis detection"""
        features = []
        
        crisis_keywords = {
            'suicide_direct': ['suicide', 'kill myself', 'end my life', 'take my life'],
            'suicide_indirect': ['better off dead', 'no point living', 'end it all'],
            'hopelessness': ['hopeless', 'no hope', 'give up', 'cant go on'],
            'worthlessness': ['worthless', 'useless', 'burden', 'waste of space'],
            'isolation': ['alone', 'nobody cares', 'no one understands', 'isolated'],
            'pain': ['unbearable', 'cant take it', 'too much pain', 'suffering'],
            'method': ['pills', 'rope', 'bridge', 'gun', 'knife', 'overdose'],
            'timeline': ['tonight', 'today', 'soon', 'planning', 'ready']
        }
        
        for text in texts:
            text_lower = text.lower()
            feature_vector = []
            
            # Count crisis indicators
            for category, keywords in crisis_keywords.items():
                count = sum(1 for keyword in keywords if keyword in text_lower)
                feature_vector.append(count)
            
            # Text statistics
            feature_vector.extend([
                len(text.split()),  # word count
                text.count('!'),    # exclamation marks
                text.count('?'),    # question marks
                len([w for w in text.lower().split() if w in ['i', 'me', 'my']]) / max(len(text.split()), 1)  # first person rate
            ])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def train(self, texts: List[str], labels: List[int]):
        """Train crisis intervention model"""
        features = self.extract_crisis_features(texts)
        self.model.fit(features, labels)
        return self.model
    
    def predict_risk(self, texts: List[str]) -> List[Dict[str, float]]:
        """Predict crisis risk with confidence scores"""
        features = self.extract_crisis_features(texts)
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)
        
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            result = {
                'risk_level': self.risk_levels[pred],
                'confidence': float(np.max(probs)),
                'risk_scores': {level: float(prob) for level, prob in zip(self.risk_levels, probs)}
            }
            results.append(result)
        
        return results

class TextModelTrainer:
    """Trainer for text-based mood and crisis models"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("mental/mental-bert-base-uncased")
        
    def prepare_bert_data(self, texts: List[str], labels: List[int], max_length: int = 512):
        """Prepare data for BERT training"""
        encodings = self.tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=max_length,
            return_tensors='pt'
        )
        
        class MoodDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
            
            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
                return item
            
            def __len__(self):
                return len(self.labels)
        
        return MoodDataset(encodings, labels)
    
    def train_bert_classifier(self, train_texts: List[str], train_labels: List[int],
                             val_texts: List[str], val_labels: List[int], epochs: int = 3):
        """Fine-tune BERT for mood classification"""
        model = BERTMoodClassifier()
        model.to(self.device)
        
        train_dataset = self.prepare_bert_data(train_texts, train_labels)
        val_dataset = self.prepare_bert_data(val_texts, val_labels)
        
        training_args = TrainingArguments(
            output_dir='./bert_mood_results',
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        trainer.train()
        return model, trainer
    
    def evaluate_model(self, model, test_texts: List[str], test_labels: List[int]) -> Dict[str, float]:
        """Evaluate model performance"""
        if isinstance(model, BERTMoodClassifier):
            # BERT evaluation
            model.eval()
            predictions = []
            
            with torch.no_grad():
                for text in test_texts:
                    encoding = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
                    output = model(encoding['input_ids'].to(self.device), 
                                 encoding['attention_mask'].to(self.device))
                    pred = torch.argmax(output, dim=1).cpu().numpy()[0]
                    predictions.append(pred)
        else:
            # Traditional ML model
            features = model.extract_crisis_features(test_texts)
            predictions = model.model.predict(features)
        
        f1 = f1_score(test_labels, predictions, average='weighted')
        report = classification_report(test_labels, predictions, output_dict=True)
        
        return {
            'f1_score': f1,
            'accuracy': report['accuracy'],
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall']
        }