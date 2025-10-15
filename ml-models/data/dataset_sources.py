import os
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import zipfile
import tarfile
from pathlib import Path

class DatasetAcquisition:
    """Acquire and manage mood analysis datasets"""
    
    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Dataset URLs and info
        self.datasets = {
            'daic_woz': {
                'url': 'https://dcapswoz.ict.usc.edu/',  # Requires registration
                'description': 'Depression detection from clinical interviews',
                'type': 'voice',
                'license': 'Research use only'
            },
            'emodb': {
                'url': 'http://emodb.bilderbar.info/download/',
                'description': 'Berlin Database of Emotional Speech',
                'type': 'voice',
                'license': 'Academic use'
            },
            'tess': {
                'url': 'https://tspace.library.utoronto.ca/handle/1807/24487',
                'description': 'Toronto Emotional Speech Set',
                'type': 'voice',
                'license': 'Creative Commons'
            },
            'reddit_depression': {
                'url': 'https://github.com/kharrigian/mental-health-datasets',
                'description': 'Reddit Self-Reported Depression Dataset',
                'type': 'text',
                'license': 'Research use'
            }
        }
    
    def download_sample_data(self):
        """Create sample datasets for development"""
        # Voice data samples
        voice_samples = {
            'file_id': [f'voice_{i:03d}' for i in range(1000)],
            'emotion': np.random.choice(['neutral', 'happy', 'sad', 'angry', 'fear'], 1000),
            'depression_level': np.random.choice(['none', 'mild', 'moderate', 'severe'], 1000),
            'speaker_id': [f'spk_{i%50:02d}' for i in range(1000)],
            'duration': np.random.uniform(2.0, 10.0, 1000)
        }
        
        voice_df = pd.DataFrame(voice_samples)
        voice_df.to_csv(self.data_dir / 'voice_metadata.csv', index=False)
        
        # Text data samples
        text_samples = {
            'post_id': [f'post_{i:04d}' for i in range(2000)],
            'text': [self._generate_sample_text() for _ in range(2000)],
            'depression_score': np.random.uniform(0, 1, 2000),
            'subreddit': np.random.choice(['depression', 'anxiety', 'mentalhealth', 'happy'], 2000),
            'timestamp': pd.date_range('2020-01-01', periods=2000, freq='H')
        }
        
        text_df = pd.DataFrame(text_samples)
        text_df.to_csv(self.data_dir / 'text_metadata.csv', index=False)
        
        print("Sample datasets created successfully")
    
    def _generate_sample_text(self) -> str:
        """Generate sample text for development"""
        templates = [
            "I've been feeling really {mood} lately. {context}",
            "Today was {mood}. {context}",
            "I can't shake this {mood} feeling. {context}",
            "Feeling {mood} about everything. {context}"
        ]
        
        moods = ['sad', 'happy', 'anxious', 'depressed', 'excited', 'worried', 'hopeful']
        contexts = [
            "Work has been stressful.",
            "My family is supportive.",
            "I'm struggling to sleep.",
            "Exercise helps me cope.",
            "Therapy is helping.",
            "I feel isolated.",
            "Friends make me smile."
        ]
        
        template = np.random.choice(templates)
        mood = np.random.choice(moods)
        context = np.random.choice(contexts)
        
        return template.format(mood=mood, context=context)
    
    def create_balanced_splits(self, df: pd.DataFrame, label_col: str, 
                             test_size: float = 0.2, val_size: float = 0.1) -> Dict[str, pd.DataFrame]:
        """Create balanced train/val/test splits"""
        from sklearn.model_selection import train_test_split
        
        # First split: train+val vs test
        train_val, test = train_test_split(
            df, test_size=test_size, stratify=df[label_col], random_state=42
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val, test_size=val_ratio, stratify=train_val[label_col], random_state=42
        )
        
        return {'train': train, 'val': val, 'test': test}
    
    def augment_voice_data(self, audio_files: List[str], labels: List[int]) -> Tuple[List[str], List[int]]:
        """Augment voice data for balanced training"""
        augmented_files = audio_files.copy()
        augmented_labels = labels.copy()
        
        # Count samples per class
        from collections import Counter
        label_counts = Counter(labels)
        max_count = max(label_counts.values())
        
        for label, count in label_counts.items():
            if count < max_count:
                # Find files with this label
                label_files = [f for f, l in zip(audio_files, labels) if l == label]
                needed = max_count - count
                
                # Duplicate files to balance
                for _ in range(needed):
                    file_to_duplicate = np.random.choice(label_files)
                    augmented_files.append(file_to_duplicate)
                    augmented_labels.append(label)
        
        return augmented_files, augmented_labels
    
    def augment_text_data(self, texts: List[str], labels: List[int]) -> Tuple[List[str], List[int]]:
        """Augment text data using paraphrasing"""
        augmented_texts = texts.copy()
        augmented_labels = labels.copy()
        
        # Simple text augmentation techniques
        augmentation_techniques = [
            self._synonym_replacement,
            self._random_insertion,
            self._random_swap
        ]
        
        from collections import Counter
        label_counts = Counter(labels)
        max_count = max(label_counts.values())
        
        for label, count in label_counts.items():
            if count < max_count:
                label_texts = [t for t, l in zip(texts, labels) if l == label]
                needed = max_count - count
                
                for _ in range(needed):
                    text_to_augment = np.random.choice(label_texts)
                    technique = np.random.choice(augmentation_techniques)
                    augmented_text = technique(text_to_augment)
                    
                    augmented_texts.append(augmented_text)
                    augmented_labels.append(label)
        
        return augmented_texts, augmented_labels
    
    def _synonym_replacement(self, text: str) -> str:
        """Replace words with synonyms"""
        # Simple synonym replacement
        synonyms = {
            'sad': 'depressed', 'happy': 'joyful', 'angry': 'furious',
            'worried': 'anxious', 'tired': 'exhausted', 'good': 'great'
        }
        
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in synonyms and np.random.random() < 0.3:
                words[i] = synonyms[word.lower()]
        
        return ' '.join(words)
    
    def _random_insertion(self, text: str) -> str:
        """Insert random words"""
        filler_words = ['really', 'very', 'quite', 'somewhat', 'rather']
        words = text.split()
        
        if len(words) > 3:
            insert_pos = np.random.randint(1, len(words))
            filler = np.random.choice(filler_words)
            words.insert(insert_pos, filler)
        
        return ' '.join(words)
    
    def _random_swap(self, text: str) -> str:
        """Swap random words"""
        words = text.split()
        
        if len(words) > 3:
            idx1, idx2 = np.random.choice(len(words), 2, replace=False)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def validate_dataset_quality(self, df: pd.DataFrame, label_col: str) -> Dict:
        """Validate dataset quality and balance"""
        quality_report = {
            'total_samples': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'label_distribution': df[label_col].value_counts().to_dict(),
            'duplicate_count': df.duplicated().sum()
        }
        
        # Check class balance
        label_counts = df[label_col].value_counts()
        balance_ratio = label_counts.min() / label_counts.max()
        quality_report['class_balance_ratio'] = balance_ratio
        quality_report['is_balanced'] = balance_ratio > 0.5
        
        return quality_report
    
    def export_dataset_info(self):
        """Export dataset information and licensing"""
        info = {
            'datasets': self.datasets,
            'usage_guidelines': {
                'academic_use': 'Cite original papers and respect licensing terms',
                'commercial_use': 'Check individual dataset licenses',
                'data_privacy': 'Ensure compliance with privacy regulations'
            },
            'citations': {
                'daic_woz': 'Gratch et al. (2014). The Distress Analysis Interview Corpus of human and computer interviews.',
                'emodb': 'Burkhardt et al. (2005). A database of German emotional speech.',
                'tess': 'Dupuis & Pichora-Fuller (2010). Toronto emotional speech set (TESS).',
                'reddit_depression': 'Harrigian et al. (2021). Mental health datasets from social media.'
            }
        }
        
        import json
        with open(self.data_dir / 'dataset_info.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        print("Dataset information exported to dataset_info.json")