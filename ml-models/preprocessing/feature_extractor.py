import numpy as np
import pandas as pd
import librosa
import re
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

class VoiceFeatureExtractor:
    """Extract prosodic and spectral features from audio"""
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.scaler = StandardScaler()
        
    def extract_prosodic_features(self, audio_path: str) -> Dict:
        """Extract pitch, jitter, shimmer, energy features"""
        y, sr = librosa.load(audio_path, sr=self.sr)
        
        # Pitch (F0) features
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        pitch_mean = np.mean(pitch_values) if pitch_values else 0
        pitch_std = np.std(pitch_values) if pitch_values else 0
        
        # Energy features
        rms = librosa.feature.rms(y=y)[0]
        energy_mean = np.mean(rms)
        energy_std = np.std(rms)
        
        # Zero crossing rate (related to jitter/shimmer)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = np.mean(zcr)
        
        return {
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
            'energy_mean': energy_mean,
            'energy_std': energy_std,
            'zcr_mean': zcr_mean
        }
    
    def extract_mfcc_features(self, audio_path: str, n_mfcc: int = 13) -> np.ndarray:
        """Extract MFCC features for spectral analysis"""
        y, sr = librosa.load(audio_path, sr=self.sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Statistical features from MFCCs
        mfcc_features = []
        for i in range(n_mfcc):
            mfcc_features.extend([
                np.mean(mfccs[i]),
                np.std(mfccs[i]),
                np.max(mfccs[i]),
                np.min(mfccs[i])
            ])
        
        return np.array(mfcc_features)
    
    def extract_tempo_features(self, audio_path: str) -> Dict:
        """Extract tempo and rhythm features"""
        y, sr = librosa.load(audio_path, sr=self.sr)
        
        # Tempo
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        # Speaking rate (approximate)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        speaking_rate = len(onset_frames) / (len(y) / sr)  # onsets per second
        
        # Pause detection (silence segments)
        intervals = librosa.effects.split(y, top_db=20)
        pause_count = len(intervals) - 1 if len(intervals) > 1 else 0
        pause_frequency = pause_count / (len(y) / sr)
        
        return {
            'tempo': tempo,
            'speaking_rate': speaking_rate,
            'pause_frequency': pause_frequency
        }
    
    def extract_all_features(self, audio_path: str) -> np.ndarray:
        """Extract all voice features"""
        prosodic = self.extract_prosodic_features(audio_path)
        mfcc = self.extract_mfcc_features(audio_path)
        tempo = self.extract_tempo_features(audio_path)
        
        # Combine all features
        features = list(prosodic.values()) + list(mfcc) + list(tempo.values())
        return np.array(features)

class TextFeatureExtractor:
    """Extract linguistic and semantic features from text"""
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = StandardScaler()
        
        # Download NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def extract_liwc_features(self, text: str) -> Dict:
        """Extract LIWC-style linguistic features"""
        words = word_tokenize(text.lower())
        total_words = len(words)
        
        if total_words == 0:
            return {f'liwc_{key}': 0 for key in ['positive', 'negative', 'anxiety', 'sadness', 'anger']}
        
        # Emotion word lists (simplified LIWC categories)
        positive_words = {'happy', 'joy', 'love', 'good', 'great', 'wonderful', 'amazing', 'excellent'}
        negative_words = {'sad', 'bad', 'terrible', 'awful', 'hate', 'angry', 'depressed', 'lonely'}
        anxiety_words = {'worry', 'anxious', 'nervous', 'scared', 'fear', 'panic', 'stress'}
        sadness_words = {'sad', 'depressed', 'down', 'blue', 'miserable', 'hopeless'}
        anger_words = {'angry', 'mad', 'furious', 'rage', 'hate', 'annoyed'}
        
        features = {
            'liwc_positive': sum(1 for word in words if word in positive_words) / total_words,
            'liwc_negative': sum(1 for word in words if word in negative_words) / total_words,
            'liwc_anxiety': sum(1 for word in words if word in anxiety_words) / total_words,
            'liwc_sadness': sum(1 for word in words if word in sadness_words) / total_words,
            'liwc_anger': sum(1 for word in words if word in anger_words) / total_words
        }
        
        return features
    
    def extract_bert_embeddings(self, texts: List[str]) -> np.ndarray:
        """Extract BERT embeddings"""
        encodings = self.tokenizer(texts, truncation=True, padding=True, 
                                 max_length=512, return_tensors='pt')
        
        # For simplicity, return tokenized input_ids as features
        # In practice, you'd use the actual BERT model to get embeddings
        return encodings['input_ids'].numpy()
    
    def extract_semantic_features(self, text: str) -> Dict:
        """Extract semantic and emotional indicators"""
        # Text statistics
        word_count = len(text.split())
        char_count = len(text)
        sentence_count = len(re.split(r'[.!?]+', text))
        avg_word_length = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Punctuation analysis
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': sentence_count,
            'avg_word_length': avg_word_length,
            'exclamation_ratio': exclamation_count / word_count if word_count > 0 else 0,
            'question_ratio': question_count / word_count if word_count > 0 else 0
        }
    
    def extract_all_features(self, text: str) -> np.ndarray:
        """Extract all text features"""
        liwc = self.extract_liwc_features(text)
        semantic = self.extract_semantic_features(text)
        
        # Combine features
        features = list(liwc.values()) + list(semantic.values())
        return np.array(features)

class DataPreprocessor:
    """Preprocess and standardize datasets"""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.voice_scaler = StandardScaler()
        self.text_scaler = StandardScaler()
        
    def standardize_labels(self, labels: List[str]) -> np.ndarray:
        """Standardize labels across datasets to app mood categories"""
        # Map various dataset labels to 5 mood categories
        mood_mapping = {
            # DAIC-WOZ, EmoDB mappings
            'neutral': 0, 'calm': 0, 'normal': 0,
            'happy': 1, 'joy': 1, 'excited': 1,
            'sad': 2, 'depressed': 2, 'down': 2,
            'angry': 3, 'frustrated': 3, 'annoyed': 3,
            'anxious': 4, 'fear': 4, 'worried': 4
        }
        
        standardized = []
        for label in labels:
            label_lower = label.lower()
            standardized.append(mood_mapping.get(label_lower, 0))  # Default to neutral
        
        return np.array(standardized)
    
    def augment_data(self, X: np.ndarray, y: np.ndarray, 
                    augmentation_factor: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
        """Augment data for balanced training"""
        from collections import Counter
        
        # Count samples per class
        class_counts = Counter(y)
        max_count = max(class_counts.values())
        
        augmented_X = []
        augmented_y = []
        
        for class_label in class_counts.keys():
            class_indices = np.where(y == class_label)[0]
            class_X = X[class_indices]
            
            # Add original samples
            augmented_X.append(class_X)
            augmented_y.extend([class_label] * len(class_X))
            
            # Add augmented samples if needed
            target_count = int(max_count * augmentation_factor)
            current_count = len(class_X)
            
            if current_count < target_count:
                needed = target_count - current_count
                
                # Simple augmentation: add noise
                for _ in range(needed):
                    idx = np.random.choice(len(class_X))
                    sample = class_X[idx].copy()
                    noise = np.random.normal(0, 0.01, sample.shape)
                    augmented_sample = sample + noise
                    
                    augmented_X.append(augmented_sample.reshape(1, -1))
                    augmented_y.append(class_label)
        
        return np.vstack(augmented_X), np.array(augmented_y)
    
    def create_multimodal_features(self, voice_features: np.ndarray, 
                                 text_features: np.ndarray) -> np.ndarray:
        """Combine voice and text features"""
        # Normalize features separately
        voice_norm = self.voice_scaler.fit_transform(voice_features)
        text_norm = self.text_scaler.fit_transform(text_features)
        
        # Concatenate features
        combined = np.hstack([voice_norm, text_norm])
        return combined