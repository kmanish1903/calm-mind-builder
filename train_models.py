#!/usr/bin/env python3
"""
Train mood analysis models
Usage: python train_models.py
"""

import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split

# Add ml-models to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ml-models'))

from training.model_trainer import ModelTrainer
from preprocessing.feature_extractor import DataPreprocessor

def generate_sample_data():
    """Generate sample training data"""
    # Voice features (60 dimensions)
    X_voice = np.random.randn(1000, 60)
    
    # Text samples
    texts = [
        "I feel great today", "This is wonderful", "I'm so happy",
        "I'm feeling down", "This is terrible", "I'm sad",
        "I'm really angry", "This makes me furious", "I hate this",
        "I'm worried about this", "This makes me anxious", "I'm scared",
        "Everything is normal", "Just another day", "Nothing special"
    ] * 67  # Repeat to get ~1000 samples
    
    # Labels (0=neutral, 1=happy, 2=sad, 3=angry, 4=anxious)
    y = np.random.randint(0, 5, 1000)
    
    return X_voice, texts[:1000], y

def main():
    print("Training mood analysis models...")
    
    # Generate sample data
    X_voice, texts, y = generate_sample_data()
    
    # Split data
    X_train, X_test, texts_train, texts_test, y_train, y_test = train_test_split(
        X_voice, texts, y, test_size=0.2, random_state=42
    )
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Train voice models
    print("Training LSTM model...")
    lstm_model = trainer.train_voice_lstm(X_train, y_train, X_test, y_test)
    
    print("Training CNN model...")
    cnn_model = trainer.train_voice_cnn(X_train, y_train, X_test, y_test)
    
    # Train text model
    print("Training BERT classifier...")
    bert_path = trainer.train_bert_classifier(texts_train, y_train.tolist())
    
    # Train crisis models
    print("Training crisis classifiers...")
    crisis_models = trainer.train_crisis_classifier(X_train, y_train)
    
    # Evaluate models
    print("\nEvaluating models...")
    lstm_results = trainer.evaluate_model(lstm_model, X_test, y_test)
    print(f"LSTM F1 Score: {lstm_results['f1_mean']:.3f}")
    
    cnn_results = trainer.evaluate_model(cnn_model, X_test, y_test)
    print(f"CNN F1 Score: {cnn_results['f1_mean']:.3f}")
    
    for name, model in crisis_models.items():
        results = trainer.evaluate_model(model, X_test, y_test)
        print(f"{name} F1 Score: {results['f1_mean']:.3f}")
    
    print("\nTraining completed! Models saved to trained_models/")

if __name__ == "__main__":
    main()