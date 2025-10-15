import numpy as np
import re
from typing import Dict, List
from transformers import AutoTokenizer, AutoModel
import torch

class TextFeatureExtractor:
    """Extract linguistic and semantic features from text"""
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        
    def extract_liwc_features(self, text: str) -> Dict[str, float]:
        """Extract LIWC-style linguistic features"""
        words = text.lower().split()
        total_words = len(words)
        
        # Emotional categories
        positive_words = {'happy', 'joy', 'love', 'good', 'great', 'amazing', 'wonderful'}
        negative_words = {'sad', 'depressed', 'angry', 'hate', 'terrible', 'awful', 'horrible'}
        anxiety_words = {'worried', 'anxious', 'nervous', 'scared', 'fear', 'panic'}
        
        # Personal pronouns
        first_person = {'i', 'me', 'my', 'mine', 'myself'}
        social_words = {'we', 'us', 'they', 'them', 'people', 'friends', 'family'}
        
        features = {
            'word_count': total_words,
            'positive_emotion': sum(1 for w in words if w in positive_words) / max(total_words, 1),
            'negative_emotion': sum(1 for w in words if w in negative_words) / max(total_words, 1),
            'anxiety': sum(1 for w in words if w in anxiety_words) / max(total_words, 1),
            'first_person': sum(1 for w in words if w in first_person) / max(total_words, 1),
            'social': sum(1 for w in words if w in social_words) / max(total_words, 1),
            'sentence_length': total_words / max(text.count('.') + text.count('!') + text.count('?'), 1)
        }
        
        return features
    
    def extract_bert_embeddings(self, texts: List[str]) -> np.ndarray:
        """Extract BERT embeddings"""
        embeddings = []
        
        for text in texts:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def extract_semantic_features(self, text: str) -> Dict[str, float]:
        """Extract semantic and emotional indicators"""
        # Depression indicators
        depression_patterns = [
            r'\b(can\'t|cannot)\b', r'\bno(thing|body|where)\b', r'\b(never|always)\b',
            r'\b(hopeless|worthless|useless)\b', r'\b(tired|exhausted|drained)\b'
        ]
        
        # Crisis indicators
        crisis_patterns = [
            r'\b(kill|die|death|suicide)\b', r'\b(end it all|give up)\b',
            r'\b(hurt myself|self harm)\b', r'\b(no point|pointless)\b'
        ]
        
        features = {
            'depression_score': sum(len(re.findall(pattern, text.lower())) for pattern in depression_patterns),
            'crisis_score': sum(len(re.findall(pattern, text.lower())) for pattern in crisis_patterns),
            'text_length': len(text),
            'exclamation_ratio': text.count('!') / max(len(text), 1),
            'question_ratio': text.count('?') / max(len(text), 1)
        }
        
        return features