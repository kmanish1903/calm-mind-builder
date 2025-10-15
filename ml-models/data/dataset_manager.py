import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import librosa
import soundfile as sf
from typing import Dict, List, Tuple, Optional
import json

class DatasetManager:
    """Manages dataset acquisition, preprocessing, and standardization"""
    
    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = data_dir
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.mood_categories = {
            'very_low': 0, 'low': 1, 'moderate': 2, 'good': 3, 'excellent': 4
        }
        
    def standardize_labels(self, dataset_labels: List[str], dataset_type: str) -> List[int]:
        """Standardize labels across different datasets to app mood categories"""
        label_mapping = {
            'daic_woz': {
                'no_depression': 2, 'mild_depression': 1, 'moderate_depression': 0,
                'severe_depression': 0
            },
            'emodb': {
                'neutral': 2, 'happy': 3, 'sad': 1, 'angry': 1, 'fear': 0, 'disgust': 1
            },
            'tess': {
                'neutral': 2, 'happy': 4, 'sad': 1, 'angry': 1, 'fear': 0, 'disgust': 1,
                'surprise': 3
            },
            'reddit_depression': {
                'not_depressed': 3, 'mild': 2, 'moderate': 1, 'severe': 0
            }
        }
        
        mapping = label_mapping.get(dataset_type, {})
        return [mapping.get(label.lower(), 2) for label in dataset_labels]
    
    def create_balanced_dataset(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create balanced training set using data augmentation"""
        from imblearn.over_sampling import SMOTE
        
        smote = SMOTE(random_state=42)
        features_balanced, labels_balanced = smote.fit_resample(features, labels)
        return features_balanced, labels_balanced
    
    def preprocess_audio_batch(self, audio_files: List[str], target_sr: int = 16000) -> np.ndarray:
        """Batch preprocess audio files"""
        processed_audio = []
        
        for file_path in audio_files:
            try:
                # Load and resample audio
                audio, sr = librosa.load(file_path, sr=target_sr)
                
                # Normalize audio
                audio = librosa.util.normalize(audio)
                
                # Trim silence
                audio, _ = librosa.effects.trim(audio, top_db=20)
                
                # Pad or truncate to fixed length (3 seconds)
                target_length = target_sr * 3
                if len(audio) > target_length:
                    audio = audio[:target_length]
                else:
                    audio = np.pad(audio, (0, target_length - len(audio)))
                
                processed_audio.append(audio)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                # Add zero array for failed files
                processed_audio.append(np.zeros(target_sr * 3))
        
        return np.array(processed_audio)
    
    def preprocess_text_batch(self, texts: List[str]) -> List[str]:
        """Batch preprocess text data"""
        import re
        
        processed_texts = []
        for text in texts:
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            # Remove special characters but keep punctuation
            text = re.sub(r'[^\w\s.,!?;:]', '', text)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            processed_texts.append(text)
        
        return processed_texts
    
    def save_preprocessed_data(self, data: Dict, filename: str):
        """Save preprocessed data to disk"""
        os.makedirs(self.data_dir, exist_ok=True)
        filepath = os.path.join(self.data_dir, filename)
        
        if filename.endswith('.npy'):
            np.save(filepath, data)
        elif filename.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            pd.DataFrame(data).to_csv(filepath, index=False)
    
    def load_preprocessed_data(self, filename: str):
        """Load preprocessed data from disk"""
        filepath = os.path.join(self.data_dir, filename)
        
        if filename.endswith('.npy'):
            return np.load(filepath)
        elif filename.endswith('.json'):
            with open(filepath, 'r') as f:
                return json.load(f)
        else:
            return pd.read_csv(filepath)
    
    def create_train_test_split(self, features: np.ndarray, labels: np.ndarray, 
                               test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create stratified train-test split"""
        return train_test_split(features, labels, test_size=test_size, 
                               stratify=labels, random_state=42)