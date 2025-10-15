#!/usr/bin/env python3
"""
Simple mood analysis that works without pre-trained models
Uses rule-based analysis and basic ML
"""

import numpy as np
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
import os

class SimpleMoodAnalyzer:
    def __init__(self):
        self.mood_classes = ['neutral', 'positive', 'negative', 'anxious', 'excited']
        self.text_classifier = None
        self._setup_classifier()
    
    def _setup_classifier(self):
        """Setup a simple text classifier"""
        # Sample training data
        training_texts = [
            # Positive
            "I feel great today", "This is wonderful", "I'm so happy", "Amazing day",
            "Love this", "Fantastic", "Excellent work", "Perfect", "Brilliant",
            # Negative  
            "I feel terrible", "This is awful", "I'm so sad", "Horrible day",
            "Hate this", "Terrible", "Bad work", "Worst", "Depressing",
            # Anxious
            "I'm worried", "This makes me nervous", "I'm scared", "Anxious about this",
            "Stressed out", "Panic", "Fearful", "Concerned", "Uneasy",
            # Excited
            "I'm so excited", "Can't wait", "Thrilled", "Pumped up",
            "Energetic", "Enthusiastic", "Eager", "Hyped", "Elated",
            # Neutral
            "It's okay", "Normal day", "Nothing special", "Average", "Fine",
            "Regular", "Standard", "Typical", "Ordinary", "Usual"
        ]
        
        training_labels = (
            [1] * 9 +  # positive
            [2] * 9 +  # negative  
            [3] * 9 +  # anxious
            [4] * 9 +  # excited
            [0] * 10   # neutral (10 to match total)
        )
        
        # Create and train classifier
        self.text_classifier = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('classifier', MultinomialNB())
        ])
        
        self.text_classifier.fit(training_texts, training_labels)
    
    def analyze_text(self, text):
        """Analyze mood from text"""
        if not text or len(text.strip()) == 0:
            return "neutral", 0.5
        
        # Get prediction
        prediction = self.text_classifier.predict([text])[0]
        probabilities = self.text_classifier.predict_proba([text])[0]
        confidence = max(probabilities)
        
        mood = self.mood_classes[prediction]
        return mood, confidence
    
    def analyze_voice_features(self, features):
        """Simulate voice analysis from features"""
        if features is None or len(features) == 0:
            return "neutral", 0.5
        
        # Simple rule-based analysis of voice features
        energy = np.mean(features[:10]) if len(features) >= 10 else 0
        pitch_var = np.std(features[10:20]) if len(features) >= 20 else 0
        
        if energy > 0.5 and pitch_var > 0.3:
            return "excited", 0.7
        elif energy < -0.5:
            return "negative", 0.6
        elif pitch_var > 0.5:
            return "anxious", 0.65
        elif energy > 0.2:
            return "positive", 0.6
        else:
            return "neutral", 0.5
    
    def combined_analysis(self, text, voice_features=None):
        """Combine text and voice analysis"""
        text_mood, text_conf = self.analyze_text(text)
        
        if voice_features is not None:
            voice_mood, voice_conf = self.analyze_voice_features(voice_features)
            
            # Weighted combination
            if text_mood == voice_mood:
                combined_mood = text_mood
                combined_conf = (text_conf + voice_conf) / 2
            else:
                # Choose higher confidence
                if text_conf > voice_conf:
                    combined_mood = text_mood
                    combined_conf = text_conf * 0.8
                else:
                    combined_mood = voice_mood
                    combined_conf = voice_conf * 0.8
            
            return {
                'mood': combined_mood,
                'confidence': combined_conf,
                'text_analysis': {'mood': text_mood, 'confidence': text_conf},
                'voice_analysis': {'mood': voice_mood, 'confidence': voice_conf}
            }
        else:
            return {
                'mood': text_mood,
                'confidence': text_conf,
                'text_analysis': {'mood': text_mood, 'confidence': text_conf}
            }

def main():
    parser = argparse.ArgumentParser(description='Simple mood analysis')
    parser.add_argument('--text', default="", help='Text input for analysis')
    parser.add_argument('--audio', help='Audio file (generates random features)')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = SimpleMoodAnalyzer()
    
    # Generate voice features if audio specified
    voice_features = None
    if args.audio:
        # Simulate voice feature extraction
        voice_features = np.random.randn(60)
        print(f"Simulated voice features from: {args.audio}")
    
    # Run analysis
    result = analyzer.combined_analysis(args.text, voice_features)
    
    # Display results
    print("\n=== MOOD ANALYSIS RESULTS ===")
    print(f"Input Text: '{args.text}'")
    print(f"Overall Mood: {result['mood']}")
    print(f"Overall Confidence: {result['confidence']:.2f}")
    
    if 'text_analysis' in result:
        print(f"Text Analysis: {result['text_analysis']['mood']} ({result['text_analysis']['confidence']:.2f})")
    
    if 'voice_analysis' in result:
        print(f"Voice Analysis: {result['voice_analysis']['mood']} ({result['voice_analysis']['confidence']:.2f})")
    
    print("Analysis completed successfully!")

if __name__ == "__main__":
    main()