import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
from dataclasses import dataclass

@dataclass
class MoodPrediction:
    """Mood prediction result"""
    mood_score: float
    confidence: float
    mood_category: str
    crisis_risk: str
    contributing_factors: Dict[str, float]
    model_weights: Dict[str, float]

class FusionEngine:
    """Fusion logic combining custom models with GPT-4o analysis"""
    
    def __init__(self):
        self.model_weights = {
            'voice_lstm': 0.25,
            'voice_cnn': 0.20,
            'text_bert': 0.30,
            'crisis_svm': 0.15,
            'gpt4o_analysis': 0.10
        }
        
        self.mood_categories = {
            0: 'very_low', 1: 'low', 2: 'moderate', 3: 'good', 4: 'excellent'
        }
        
        self.crisis_levels = {
            0: 'low', 1: 'moderate', 2: 'high', 3: 'critical'
        }
    
    def fuse_predictions(self, model_outputs: Dict[str, Any], 
                        gpt4o_analysis: Optional[Dict[str, Any]] = None) -> MoodPrediction:
        """Combine predictions from multiple models"""
        
        # Extract individual predictions
        voice_predictions = self._process_voice_predictions(model_outputs)
        text_predictions = self._process_text_predictions(model_outputs)
        crisis_prediction = self._process_crisis_prediction(model_outputs)
        
        # Weighted fusion
        mood_scores = []
        confidences = []
        
        # Voice models
        if 'voice_lstm' in model_outputs:
            lstm_pred = model_outputs['voice_lstm']
            mood_scores.append(lstm_pred['prediction'] * self.model_weights['voice_lstm'])
            confidences.append(lstm_pred['confidence'] * self.model_weights['voice_lstm'])
        
        if 'voice_cnn' in model_outputs:
            cnn_pred = model_outputs['voice_cnn']
            mood_scores.append(cnn_pred['prediction'] * self.model_weights['voice_cnn'])
            confidences.append(cnn_pred['confidence'] * self.model_weights['voice_cnn'])
        
        # Text model
        if 'text_bert' in model_outputs:
            bert_pred = model_outputs['text_bert']
            mood_scores.append(bert_pred['prediction'] * self.model_weights['text_bert'])
            confidences.append(bert_pred['confidence'] * self.model_weights['text_bert'])
        
        # Crisis model influence
        if crisis_prediction['risk_level'] in ['high', 'critical']:
            # Adjust mood score downward for high crisis risk
            crisis_adjustment = -1.0 if crisis_prediction['risk_level'] == 'critical' else -0.5
            mood_scores.append(crisis_adjustment * self.model_weights['crisis_svm'])
        
        # GPT-4o analysis integration
        if gpt4o_analysis:
            gpt_mood = self._parse_gpt4o_mood(gpt4o_analysis)
            mood_scores.append(gpt_mood * self.model_weights['gpt4o_analysis'])
            confidences.append(0.8 * self.model_weights['gpt4o_analysis'])  # Fixed confidence for GPT
        
        # Calculate final scores
        final_mood_score = sum(mood_scores)
        final_confidence = sum(confidences)
        
        # Normalize to 0-4 range
        final_mood_score = max(0, min(4, final_mood_score))
        final_confidence = max(0, min(1, final_confidence))
        
        # Determine mood category
        mood_category = self.mood_categories[round(final_mood_score)]
        
        # Contributing factors analysis
        contributing_factors = self._analyze_contributing_factors(
            model_outputs, gpt4o_analysis
        )
        
        return MoodPrediction(
            mood_score=final_mood_score,
            confidence=final_confidence,
            mood_category=mood_category,
            crisis_risk=crisis_prediction['risk_level'],
            contributing_factors=contributing_factors,
            model_weights=self.model_weights
        )
    
    def _process_voice_predictions(self, model_outputs: Dict[str, Any]) -> Dict[str, float]:
        """Process voice model predictions"""
        voice_results = {}
        
        for model_name in ['voice_lstm', 'voice_cnn']:
            if model_name in model_outputs:
                pred = model_outputs[model_name]
                voice_results[model_name] = {
                    'prediction': pred.get('prediction', 2),
                    'confidence': pred.get('confidence', 0.5)
                }
        
        return voice_results
    
    def _process_text_predictions(self, model_outputs: Dict[str, Any]) -> Dict[str, float]:
        """Process text model predictions"""
        text_results = {}
        
        if 'text_bert' in model_outputs:
            pred = model_outputs['text_bert']
            text_results['bert'] = {
                'prediction': pred.get('prediction', 2),
                'confidence': pred.get('confidence', 0.5)
            }
        
        return text_results
    
    def _process_crisis_prediction(self, model_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process crisis intervention prediction"""
        if 'crisis_svm' in model_outputs:
            crisis_pred = model_outputs['crisis_svm']
            return {
                'risk_level': crisis_pred.get('risk_level', 'low'),
                'confidence': crisis_pred.get('confidence', 0.5),
                'risk_scores': crisis_pred.get('risk_scores', {})
            }
        
        return {'risk_level': 'low', 'confidence': 0.5, 'risk_scores': {}}
    
    def _parse_gpt4o_mood(self, gpt4o_analysis: Dict[str, Any]) -> float:
        """Parse GPT-4o analysis to extract mood score"""
        # Extract mood indicators from GPT-4o response
        mood_indicators = gpt4o_analysis.get('mood_indicators', {})
        
        # Simple mapping based on GPT analysis
        positive_indicators = mood_indicators.get('positive', 0)
        negative_indicators = mood_indicators.get('negative', 0)
        
        # Convert to 0-4 scale
        if negative_indicators > positive_indicators:
            return max(0, 2 - negative_indicators)
        else:
            return min(4, 2 + positive_indicators)
    
    def _analyze_contributing_factors(self, model_outputs: Dict[str, Any], 
                                    gpt4o_analysis: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze factors contributing to mood prediction"""
        factors = {}
        
        # Voice factors
        if 'voice_lstm' in model_outputs:
            voice_features = model_outputs['voice_lstm'].get('features', {})
            factors['vocal_energy'] = voice_features.get('energy_mean', 0.5)
            factors['speech_rate'] = voice_features.get('speaking_rate', 0.5)
            factors['vocal_stability'] = 1 - voice_features.get('jitter', 0.5)
        
        # Text factors
        if 'text_bert' in model_outputs:
            text_features = model_outputs['text_bert'].get('features', {})
            factors['emotional_language'] = text_features.get('negative_emotion_rate', 0)
            factors['social_connection'] = text_features.get('social_connection', 0)
            factors['cognitive_processing'] = text_features.get('cognitive_rate', 0.5)
        
        # Crisis factors
        if 'crisis_svm' in model_outputs:
            crisis_features = model_outputs['crisis_svm'].get('features', {})
            factors['crisis_indicators'] = crisis_features.get('crisis_indicators', 0)
            factors['hopelessness'] = crisis_features.get('hopelessness', 0)
        
        # GPT-4o factors
        if gpt4o_analysis:
            factors['contextual_understanding'] = gpt4o_analysis.get('context_score', 0.5)
            factors['emotional_nuance'] = gpt4o_analysis.get('nuance_score', 0.5)
        
        return factors
    
    def adaptive_weighting(self, model_confidences: Dict[str, float]) -> Dict[str, float]:
        """Adaptively adjust model weights based on confidence scores"""
        adaptive_weights = self.model_weights.copy()
        
        # Increase weight for high-confidence models
        total_confidence = sum(model_confidences.values())
        
        if total_confidence > 0:
            for model_name, confidence in model_confidences.items():
                if model_name in adaptive_weights:
                    # Boost weight for confident models
                    confidence_boost = (confidence - 0.5) * 0.1  # Max 5% boost
                    adaptive_weights[model_name] += confidence_boost
        
        # Normalize weights
        total_weight = sum(adaptive_weights.values())
        if total_weight > 0:
            adaptive_weights = {k: v/total_weight for k, v in adaptive_weights.items()}
        
        return adaptive_weights
    
    def create_explanation(self, prediction: MoodPrediction) -> Dict[str, Any]:
        """Create human-readable explanation of prediction"""
        explanation = {
            'mood_assessment': f"Predicted mood: {prediction.mood_category} (score: {prediction.mood_score:.2f})",
            'confidence_level': f"Confidence: {prediction.confidence:.1%}",
            'crisis_assessment': f"Crisis risk: {prediction.crisis_risk}",
            'key_factors': []
        }
        
        # Identify top contributing factors
        sorted_factors = sorted(
            prediction.contributing_factors.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        for factor, value in sorted_factors[:3]:
            impact = "positive" if value > 0.5 else "negative"
            explanation['key_factors'].append({
                'factor': factor.replace('_', ' ').title(),
                'impact': impact,
                'strength': abs(value - 0.5) * 2  # Normalize to 0-1
            })
        
        return explanation
    
    def get_recommendations(self, prediction: MoodPrediction) -> List[str]:
        """Generate recommendations based on prediction"""
        recommendations = []
        
        if prediction.crisis_risk in ['high', 'critical']:
            recommendations.extend([
                "Immediate professional support recommended",
                "Contact crisis helpline: 988 (US) or local emergency services",
                "Reach out to trusted friend or family member"
            ])
        
        elif prediction.mood_category in ['very_low', 'low']:
            recommendations.extend([
                "Consider speaking with a mental health professional",
                "Engage in gentle physical activity",
                "Practice mindfulness or meditation",
                "Connect with supportive friends or family"
            ])
        
        elif prediction.mood_category == 'moderate':
            recommendations.extend([
                "Maintain current self-care routines",
                "Monitor mood patterns",
                "Consider stress management techniques"
            ])
        
        else:  # good or excellent
            recommendations.extend([
                "Continue current positive practices",
                "Share positivity with others",
                "Maintain healthy lifestyle habits"
            ])
        
        return recommendations