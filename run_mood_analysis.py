#!/usr/bin/env python3
"""
Simple rule-based mood analysis script
Usage: python run_mood_analysis.py --text "your text here"
"""

import argparse
import re

def analyze_text_mood(text):
    """Enhanced rule-based text mood analysis"""
    if not text:
        return {'mood': 'neutral', 'confidence': 0.5, 'keywords': []}
    
    text_lower = text.lower()
    
    # Crisis/severe negative indicators
    crisis_words = ['die', 'death', 'kill', 'suicide', 'suicidal', 'end it', 'give up', 'hopeless', 'worthless', 'useless',
                   'no point', 'cant take', 'end my life', 'kill myself', 'better off dead', 'want to disappear',
                   'no reason to live', 'tired of living', 'done with life', 'self harm', 'cut myself', 'overdose',
                   'cant do it anymore', 'done with everything', 'give up completely', 'surrender to the dark',
                   'no fight left', 'letting go now', 'goodbye forever', 'over now', 'ready to quit', 'time is up',
                   'checking out', 'need to disappear', 'want to vanish', 'closing my eyes', 'done existing',
                   'wont be around', 'be gone soon', 'just let me die', 'no one will miss me', 'drain on everyone',
                   'leaving soon', 'saying my last words', 'made my decision', 'have a plan ready', 'looking for a way',
                   'gathered the things', 'finalizing my plans', 'preparing to go', 'arranging my affairs',
                   'cant wait to be at peace', 'want eternal rest', 'tired of the pain', 'need the suffering to stop',
                   'cant bear this existence', 'feel terminal', 'irreparable', 'beyond saving', 'completely hopeless',
                   'utterly finished', 'broken beyond repair', 'take my own life', 'sleep forever', 'exit plan',
                   'gonna do it', 'time to go', 'cant go on', 'saying goodbye', 'had enough', 'whats the point anymore',
                   'my absence would be better', 'dont belong here', 'pills', 'rope', 'jump', 'weapon', 'bleeding',
                   'car crash', 'bridge', 'nothing left', 'irredeemable', 'no escape', 'terminal', 'want peace']
    
    # Negative emotional words
    negative_words = ['sad', 'angry', 'frustrated', 'terrible', 'awful', 'hate', 'depressed', 'anxious', 'worried', 'bad',
                     'cry', 'cried', 'crying', 'hurt', 'pain', 'lonely', 'empty', 'broken', 'devastated', 'miserable',
                     'stressed', 'overwhelmed', 'exhausted', 'drained', 'numb', 'lost', 'confused', 'scared', 'afraid',
                     'panic', 'nightmare', 'trauma', 'betrayed', 'rejected', 'abandoned', 'isolated', 'helpless',
                     'defeated', 'crushed', 'shattered', 'destroyed', 'ruined', 'failed', 'failure', 'disappointed',
                     'regret', 'guilt', 'shame', 'embarrassed', 'humiliated', 'disgusted', 'sick', 'tired', 'weak',
                     'pathetic', 'stupid', 'ugly', 'fat', 'loser', 'burden', 'annoying', 'irritated', 'furious',
                     'rage', 'bitter', 'resentful', 'jealous', 'envious', 'insecure', 'self-doubt', 'doubt',
                     'dejected', 'crestfallen', 'despondent', 'melancholy', 'gloomy', 'withdrawn', 'agitated', 'shaky',
                     'restless', 'on edge', 'distressed', 'dismayed', 'sullen', 'vexed', 'pessimistic', 'distraught',
                     'disheartened', 'weary', 'languid', 'lethargic', 'apathetic', 'listless', 'moody', 'irritable',
                     'cranky', 'cross', 'grouchy', 'sulky', 'cheerless', 'joyless', 'sorrowful', 'heartbroken',
                     'mournful', 'desolate', 'anguished', 'tormented', 'troubled', 'afflicted', 'disquieted',
                     'perturbed', 'flustered', 'rattled', 'unnerved', 'apprehensive', 'jumpy', 'nervous', 'tense',
                     'strained', 'jittery', 'edgy', 'highly-strung', 'oversensitive', 'fragile', 'vulnerable',
                     'defensive', 'protective', 'guarded', 'secretive', 'avoidant', 'hesitant', 'reluctant',
                     'reserved', 'detached', 'reclusive', 'estranged', 'forlorn', 'hollow', 'vacant', 'dead inside',
                     'robotic', 'mechanical', 'unfeeling', 'bruised', 'scarred', 'wounded', 'aching', 'raw', 'tender',
                     'exposed', 'shivering', 'cringing', 'fearful', 'terrified', 'petrified', 'horror-struck',
                     'traumatized', 'shell-shocked', 'stunned', 'bewildered', 'perplexed', 'baffled', 'muddled',
                     'foggy', 'fuzzy', 'unclear', 'uncertain', 'irresolute', 'wavering', 'conflicted', 'divided',
                     'self-loathing', 'self-critical', 'unforgiving', 'relentless', 'guilt-ridden', 'shame-filled',
                     'mortified', 'disgraced', 'dishonored', 'sullied', 'tainted', 'corrupted', 'filthy', 'dirty',
                     'defiled', 'debased', 'degraded', 'despicable', 'contemptible', 'vile', 'nasty', 'atrocious',
                     'appalling', 'horrendous', 'dreadful', 'horrific', 'grim', 'bleak', 'dark', 'cavernous',
                     'bottomless', 'endless', 'perpetual', 'recurring', 'chronic', 'persistent', 'unyielding',
                     'unceasing', 'incompetent', 'unskilled', 'talentless', 'clumsy', 'mess', 'monster', 'evil',
                     'selfish', 'detestable', 'repulsive', 'unworthy', 'shameful', 'abused', 'manipulated',
                     'screwed over', 'deserted', 'unwanted', 'grief', 'shunned', 'mocked', 'ostracized', 'alienated',
                     'unlovable', 'inadequate', 'misunderstood', 'annoyed', 'infuriated', 'exasperated', 'provoked',
                     'incensed', 'indignant', 'scornful', 'cynical', 'skeptical', 'nihilistic', 'desperate',
                     'wretched', 'pitiful', 'pitiable', 'lamentable', 'woeful', 'lugubrious', 'morose', 'dour',
                     'taciturn', 'silent', 'speechless', 'mute', 'quiet', 'hushed', 'subdued', 'meek', 'mild',
                     'timid', 'shy', 'introverted', 'cautious', 'wary', 'suspicious', 'mistrustful', 'paranoid',
                     'dread-filled', 'foreboding', 'ominous', 'sombre', 'grieving', 'sorrowing', 'lamenting',
                     'weeping', 'sobbing', 'wailing', 'sniffling', 'whimpering', 'choked up', 'tongue-tied',
                     'oppressed', 'suppressed', 'suffocated', 'captive', 'imprisoned', 'trapped', 'constrained',
                     'restricted', 'curtailed', 'limited', 'hindered', 'hampered', 'obstructed', 'thwarted',
                     'inundated', 'submerged', 'engulfed', 'consumed', 'devoured', 'depleted', 'sapped', 'weakened',
                     'faint', 'dizzy', 'sickly', 'ill', 'unwell', 'ailing', 'suffering', 'struggling', 'utterly spent',
                     'completely drained', 'bone-tired', 'emotionally bankrupt', 'spiritually depleted', 'running on empty',
                     'out of fuel', 'cant cope', 'falling apart', 'need a break', 'at my limit']
    
    # Positive words
    positive_words = ['happy', 'joy', 'excited', 'great', 'amazing', 'wonderful', 'love', 'fantastic', 'excellent', 'good',
                     'blessed', 'grateful', 'peaceful', 'content', 'proud', 'hopeful', 'optimistic']
    
    # Check for crisis indicators first
    crisis_count = sum(1 for word in crisis_words if word in text_lower)
    if crisis_count > 0:
        return {
            'mood': 'crisis',
            'confidence': 0.95,
            'keywords': [word for word in crisis_words if word in text_lower],
            'crisis_indicators': crisis_count
        }
    
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    found_keywords = []
    for word in positive_words + negative_words:
        if word in text_lower:
            found_keywords.append(word)
    
    if pos_count > neg_count:
        mood = 'positive'
        confidence = min(0.9, 0.6 + (pos_count - neg_count) * 0.1)
    elif neg_count > pos_count:
        mood = 'negative'
        confidence = min(0.9, 0.6 + (neg_count - pos_count) * 0.1)
    else:
        mood = 'neutral'
        confidence = 0.5
    
    return {
        'mood': mood,
        'confidence': confidence,
        'keywords': found_keywords,
        'positive_count': pos_count,
        'negative_count': neg_count
    }

def main():
    parser = argparse.ArgumentParser(description='Run simple mood analysis')
    parser.add_argument('--text', required=True, help='Text input for analysis')
    
    args = parser.parse_args()
    
    # Analyze mood
    result = analyze_text_mood(args.text)
    
    # Display results
    print("\n=== MOOD ANALYSIS RESULTS ===")
    print(f"Input Text: {args.text}")
    print(f"Predicted Mood: {result['mood']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Keywords Found: {', '.join(result['keywords']) if result['keywords'] else 'None'}")
    
    if result['mood'] == 'crisis':
        print(f"WARNING: CRISIS INDICATORS DETECTED: {result['crisis_indicators']}")
        print("Please seek immediate help if you're having thoughts of self-harm.")
    else:
        print(f"Positive Words: {result.get('positive_count', 0)}")
        print(f"Negative Words: {result.get('negative_count', 0)}")

if __name__ == "__main__":
    main()