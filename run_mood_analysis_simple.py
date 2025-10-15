#!/usr/bin/env python3
"""
Simple mood analysis script that works without ML dependencies
Usage: python run_mood_analysis_simple.py --text "your text here"
"""

import argparse
import random

def analyze_text_mood(text):
    """Simple rule-based mood analysis"""
    if not text:
        return "neutral", 0.5
    
    text_lower = text.lower()
    
    # Positive keywords
    positive_words = ['happy', 'great', 'good', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'joy', 'excited']
    # Negative keywords  
    negative_words = ['sad', 'bad', 'terrible', 'awful', 'hate', 'angry', 'depressed', 'upset', 'worried', 'anxious']
    
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        return "positive", min(0.6 + (positive_count * 0.1), 0.95)
    elif negative_count > positive_count:
        return "negative", min(0.6 + (negative_count * 0.1), 0.95)
    else:
        return "neutral", 0.5

def main():
    parser = argparse.ArgumentParser(description='Run simple mood analysis')
    parser.add_argument('--text', default="", help='Text input for analysis')
    parser.add_argument('--audio', help='Audio file (simulated)')
    
    args = parser.parse_args()
    
    # Analyze text
    mood, confidence = analyze_text_mood(args.text)
    
    # Simulate voice analysis if audio provided
    voice_mood = "neutral"
    voice_confidence = 0.5
    if args.audio:
        voice_moods = ["positive", "negative", "neutral"]
        voice_mood = random.choice(voice_moods)
        voice_confidence = random.uniform(0.4, 0.9)
    
    # Display results
    print("\n=== MOOD ANALYSIS RESULTS ===")
    print(f"Input Text: '{args.text}'")
    print(f"Text Mood: {mood} (confidence: {confidence:.2f})")
    
    if args.audio:
        print(f"Audio File: {args.audio}")
        print(f"Voice Mood: {voice_mood} (confidence: {voice_confidence:.2f})")
        
        # Combined analysis
        combined_confidence = (confidence + voice_confidence) / 2
        if mood == voice_mood:
            combined_mood = mood
        else:
            combined_mood = "mixed"
        
        print(f"Combined Analysis: {combined_mood} (confidence: {combined_confidence:.2f})")
    
    print(f"Analysis completed successfully!")

if __name__ == "__main__":
    main()