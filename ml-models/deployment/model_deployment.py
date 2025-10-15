import os
import json
import numpy as np
import onnx
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from supabase import create_client, Client
from typing import Dict, List, Optional, Union
import joblib

class ModelConverter:
    """Convert models to deployment formats"""
    
    def __init__(self, model_dir: str = "trained_models"):
        self.model_dir = model_dir
        
    def pytorch_to_onnx(self, model: torch.nn.Module, input_shape: tuple, 
                       output_path: str) -> str:
        """Convert PyTorch model to ONNX format"""
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        return output_path
    
    def convert_all_models(self) -> Dict[str, str]:
        """Convert all trained models to deployment format"""
        converted_models = {}
        
        # Convert voice models
        for model_name in ['voice_lstm', 'voice_cnn']:
            model_path = os.path.join(self.model_dir, f'{model_name}.pth')
            if os.path.exists(model_path):
                # Load model architecture (simplified)
                if model_name == 'voice_lstm':
                    from ml_models.training.model_trainer import VoiceMoodLSTM
                    model = VoiceMoodLSTM(input_size=60)  # Adjust based on features
                    input_shape = (10, 6)  # seq_len, input_size
                else:
                    from ml_models.training.model_trainer import VoiceMoodCNN
                    model = VoiceMoodCNN(input_size=60)
                    input_shape = (60,)
                
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                
                onnx_path = os.path.join(self.model_dir, f'{model_name}.onnx')
                self.pytorch_to_onnx(model, input_shape, onnx_path)
                converted_models[model_name] = onnx_path
        
        return converted_models

class SupabaseDeployer:
    """Deploy models to Supabase Edge Functions"""
    
    def __init__(self, supabase_url: str, supabase_key: str):
        self.supabase: Client = create_client(supabase_url, supabase_key)
        
    def create_edge_function(self, function_name: str, model_path: str) -> str:
        """Create Supabase Edge Function for model inference"""
        
        # Edge Function code template
        edge_function_code = f"""
import {{ serve }} from "https://deno.land/std@0.168.0/http/server.ts"
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"

const modelPath = "./{os.path.basename(model_path)}"

serve(async (req) => {{
  if (req.method !== 'POST') {{
    return new Response('Method not allowed', {{ status: 405 }})
  }}
  
  try {{
    const {{ features }} = await req.json()
    
    // Load ONNX model
    const session = await ort.InferenceSession.create(modelPath)
    
    // Prepare input tensor
    const inputTensor = new ort.Tensor('float32', features, [1, features.length])
    
    // Run inference
    const results = await session.run({{ input: inputTensor }})
    const predictions = results.output.data
    
    return new Response(
      JSON.stringify({{ predictions: Array.from(predictions) }}),
      {{ headers: {{ "Content-Type": "application/json" }} }}
    )
  }} catch (error) {{
    return new Response(
      JSON.stringify({{ error: error.message }}),
      {{ status: 500, headers: {{ "Content-Type": "application/json" }} }}
    )
  }}
}})
"""
        
        # Save edge function
        function_dir = f"supabase/functions/{function_name}"
        os.makedirs(function_dir, exist_ok=True)
        
        with open(f"{function_dir}/index.ts", "w") as f:
            f.write(edge_function_code)
        
        return function_dir
    
    def deploy_models(self, model_paths: Dict[str, str]) -> Dict[str, str]:
        """Deploy all models as Edge Functions"""
        deployed_functions = {}
        
        for model_name, model_path in model_paths.items():
            function_name = f"mood-{model_name.replace('_', '-')}"
            function_dir = self.create_edge_function(function_name, model_path)
            deployed_functions[model_name] = function_name
        
        return deployed_functions

class ModelFusion:
    """Fusion logic for combining multiple models"""
    
    def __init__(self, model_weights: Optional[Dict[str, float]] = None):
        self.model_weights = model_weights or {
            'voice_lstm': 0.3,
            'voice_cnn': 0.2,
            'bert_classifier': 0.3,
            'crisis_svm': 0.1,
            'crisis_random_forest': 0.1
        }
    
    def weighted_fusion(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine predictions using weighted average"""
        weighted_preds = []
        
        for model_name, pred in predictions.items():
            weight = self.model_weights.get(model_name, 0.2)
            weighted_preds.append(pred * weight)
        
        return np.sum(weighted_preds, axis=0)
    
    def ensemble_fusion(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine predictions using ensemble voting"""
        # Convert probabilities to class predictions
        class_preds = []
        for pred in predictions.values():
            class_preds.append(np.argmax(pred, axis=1))
        
        # Majority voting
        class_preds = np.array(class_preds).T
        ensemble_pred = []
        
        for sample_preds in class_preds:
            unique, counts = np.unique(sample_preds, return_counts=True)
            ensemble_pred.append(unique[np.argmax(counts)])
        
        return np.array(ensemble_pred)
    
    def gpt4_fusion(self, predictions: Dict[str, np.ndarray], 
                   text_input: str, voice_features: np.ndarray) -> Dict:
        """Combine custom models with GPT-4o analysis"""
        # Get ensemble prediction
        ensemble_pred = self.weighted_fusion(predictions)
        mood_classes = ['neutral', 'happy', 'sad', 'angry', 'anxious']
        predicted_mood = mood_classes[np.argmax(ensemble_pred)]
        confidence = np.max(ensemble_pred)
        
        # Prepare context for GPT-4o
        context = {
            'predicted_mood': predicted_mood,
            'confidence': float(confidence),
            'text_input': text_input,
            'voice_analysis': {
                'energy_level': float(voice_features[2]) if len(voice_features) > 2 else 0,
                'pitch_variation': float(voice_features[1]) if len(voice_features) > 1 else 0
            },
            'model_predictions': {
                name: pred.tolist() for name, pred in predictions.items()
            }
        }
        
        return context

class InferenceAPI:
    """API endpoints for model inference"""
    
    def __init__(self, model_dir: str = "trained_models"):
        self.model_dir = model_dir
        self.models = self._load_models()
        self.fusion = ModelFusion()
        
    def _load_models(self) -> Dict:
        """Load all trained models"""
        models = {}
        
        # Load sklearn models
        for model_name in ['crisis_svm', 'crisis_random_forest']:
            model_path = os.path.join(self.model_dir, f'{model_name}.pkl')
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)
        
        # Load BERT model
        bert_path = os.path.join(self.model_dir, 'bert_classifier')
        if os.path.exists(bert_path):
            models['bert_tokenizer'] = AutoTokenizer.from_pretrained(bert_path)
            models['bert_model'] = AutoModelForSequenceClassification.from_pretrained(bert_path)
        
        return models
    
    def predict_mood(self, voice_features: np.ndarray, 
                    text_input: str) -> Dict:
        """Predict mood from voice and text inputs"""
        predictions = {}
        
        # Text-based predictions
        if 'bert_model' in self.models:
            tokenizer = self.models['bert_tokenizer']
            model = self.models['bert_model']
            
            inputs = tokenizer(text_input, return_tensors='pt', 
                             truncation=True, padding=True, max_length=512)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                predictions['bert_classifier'] = probs.numpy()
        
        # Crisis intervention scoring
        combined_features = np.concatenate([voice_features, [len(text_input.split())]])
        combined_features = combined_features.reshape(1, -1)
        
        for model_name in ['crisis_svm', 'crisis_random_forest']:
            if model_name in self.models:
                pred = self.models[model_name].predict_proba(combined_features)
                predictions[model_name] = pred
        
        # Fusion with GPT-4o context
        if predictions:
            result = self.fusion.gpt4_fusion(predictions, text_input, voice_features)
            return result
        
        return {'error': 'No models available for prediction'}
    
    def health_check(self) -> Dict:
        """Check model availability"""
        return {
            'models_loaded': list(self.models.keys()),
            'status': 'healthy' if self.models else 'no_models'
        }

class FallbackHandler:
    """Handle model availability and fallbacks"""
    
    def __init__(self):
        self.fallback_responses = {
            'neutral': {'mood': 'neutral', 'confidence': 0.5, 'source': 'fallback'},
            'error': {'mood': 'unknown', 'confidence': 0.0, 'source': 'error'}
        }
    
    def handle_model_failure(self, error_type: str) -> Dict:
        """Provide fallback response when models fail"""
        if error_type == 'network_error':
            return self.fallback_responses['neutral']
        elif error_type == 'model_error':
            return self.fallback_responses['error']
        else:
            return self.fallback_responses['neutral']
    
    def validate_input(self, voice_features: np.ndarray, 
                      text_input: str) -> bool:
        """Validate input data"""
        if voice_features is None or len(voice_features) == 0:
            return False
        if text_input is None or len(text_input.strip()) == 0:
            return False
        return True