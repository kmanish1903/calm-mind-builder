import numpy as np
import librosa
from typing import Dict, List, Tuple
import scipy.stats as stats

class VoiceFeatureExtractor:
    """Extract prosodic and spectral features from voice recordings"""
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
        
    def extract_prosodic_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract prosodic features: pitch, jitter, shimmer, energy"""
        features = {}
        
        # Pitch (F0) features
        f0 = librosa.yin(audio, fmin=50, fmax=400, sr=self.sr)
        f0_clean = f0[f0 > 0]  # Remove unvoiced frames
        
        if len(f0_clean) > 0:
            features['f0_mean'] = np.mean(f0_clean)
            features['f0_std'] = np.std(f0_clean)
            features['f0_min'] = np.min(f0_clean)
            features['f0_max'] = np.max(f0_clean)
            features['f0_range'] = features['f0_max'] - features['f0_min']
        else:
            features.update({
                'f0_mean': 0, 'f0_std': 0, 'f0_min': 0, 'f0_max': 0, 'f0_range': 0
            })
        
        # Jitter (pitch variation)
        if len(f0_clean) > 1:
            f0_diff = np.diff(f0_clean)
            features['jitter'] = np.mean(np.abs(f0_diff)) / np.mean(f0_clean) if np.mean(f0_clean) > 0 else 0
        else:
            features['jitter'] = 0
        
        # Shimmer (amplitude variation)
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        if len(rms) > 1:
            rms_diff = np.diff(rms)
            features['shimmer'] = np.mean(np.abs(rms_diff)) / np.mean(rms) if np.mean(rms) > 0 else 0
        else:
            features['shimmer'] = 0
        
        # Energy features
        features['energy_mean'] = np.mean(rms)
        features['energy_std'] = np.std(rms)
        features['energy_max'] = np.max(rms)
        
        return features
    
    def extract_mfcc_features(self, audio: np.ndarray, n_mfcc: int = 13) -> Dict[str, float]:
        """Extract MFCC features for spectral analysis"""
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=n_mfcc)
        
        features = {}
        for i in range(n_mfcc):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
            features[f'mfcc_{i}_skew'] = stats.skew(mfccs[i])
            features[f'mfcc_{i}_kurtosis'] = stats.kurtosis(mfccs[i])
        
        return features
    
    def extract_tempo_rhythm_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract tempo and rhythm features"""
        features = {}
        
        # Speaking rate estimation
        onset_frames = librosa.onset.onset_detect(y=audio, sr=self.sr)
        duration = len(audio) / self.sr
        features['speaking_rate'] = len(onset_frames) / duration if duration > 0 else 0
        
        # Pause detection
        rms = librosa.feature.rms(y=audio)[0]
        silence_threshold = np.percentile(rms, 20)
        silence_frames = rms < silence_threshold
        
        # Count pause segments
        pause_segments = []
        in_pause = False
        pause_start = 0
        
        for i, is_silent in enumerate(silence_frames):
            if is_silent and not in_pause:
                in_pause = True
                pause_start = i
            elif not is_silent and in_pause:
                in_pause = False
                pause_length = (i - pause_start) * 512 / self.sr  # Convert to seconds
                if pause_length > 0.1:  # Only count pauses > 100ms
                    pause_segments.append(pause_length)
        
        features['pause_frequency'] = len(pause_segments) / duration if duration > 0 else 0
        features['avg_pause_duration'] = np.mean(pause_segments) if pause_segments else 0
        features['total_pause_time'] = np.sum(pause_segments) if pause_segments else 0
        features['speech_ratio'] = 1 - (features['total_pause_time'] / duration) if duration > 0 else 0
        
        # Rhythm regularity
        if len(onset_frames) > 1:
            onset_intervals = np.diff(onset_frames) * 512 / self.sr
            features['rhythm_regularity'] = 1 / (np.std(onset_intervals) + 1e-6)
        else:
            features['rhythm_regularity'] = 0
        
        return features
    
    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract additional spectral features"""
        features = {}
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        return features
    
    def extract_all_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract all voice features"""
        all_features = {}
        
        all_features.update(self.extract_prosodic_features(audio))
        all_features.update(self.extract_mfcc_features(audio))
        all_features.update(self.extract_tempo_rhythm_features(audio))
        all_features.update(self.extract_spectral_features(audio))
        
        return all_features
    
    def create_feature_pipeline(self, audio_batch: List[np.ndarray]) -> np.ndarray:
        """Create feature extraction pipeline for batch processing"""
        feature_vectors = []
        
        for audio in audio_batch:
            features = self.extract_all_features(audio)
            feature_vector = list(features.values())
            feature_vectors.append(feature_vector)
        
        return np.array(feature_vectors)
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features for model training"""
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        return scaler.fit_transform(features)