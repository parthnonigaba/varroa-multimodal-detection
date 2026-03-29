"""
Audio Health Monitor - Bee colony health detection via audio analysis
Updated to load pickle-based models (compatible with Colab-trained models)
"""

import os
import subprocess
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import json
import pickle

import numpy as np
import librosa
import logging

# Try to import tensorflow, but don't fail if not available
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False


class AudioHealthMonitor:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.logger = logging.getLogger(__name__)
        a = config.get("audio", {})
        self.device = a.get("device", "hw:2,0")
        self.sample_rate = int(a.get("sample_rate", 48000))
        self.channels = int(a.get("channels", 2))
        self.sample_format = a.get("format", "S32_LE")
        self.rec_duration = int(a.get("recording_duration", 30))
        self.rec_interval = int(a.get("recording_interval", 600))
        
        # Model paths - support both old .h5 path and new pickle paths
        self.model_path = a.get("model_path")  # Legacy .h5 path
        
        # New pickle-based model paths
        self.weights_path = a.get("weights_path", "models/audio_classifier/bee_audio_weights.pkl")
        self.architecture_path = a.get("architecture_path", "models/audio_classifier/bee_audio_architecture.json")
        self.scaler_path = a.get("scaler_path", "models/audio_classifier/audio_scaler.pkl")
        self.label_encoder_path = a.get("label_encoder_path", "models/audio_classifier/audio_label_encoder.pkl")
        
        self.unhealthy_label = str(a.get("unhealthy_label", "unhealthy"))
        self.unhealthy_threshold = float(a.get("unhealthy_threshold", 0.7))
        self.audio_dir = config.get("storage", {}).get("audio_dir", "data/audio")
        os.makedirs(self.audio_dir, exist_ok=True)

        # Model components
        self._model = None
        self._scaler = None
        self._label_encoder = None
        self._weights = None
        self._architecture = None
        
        # Map numeric class index → human label (fallback)
        self._labels = {0: "healthy", 1: "unhealthy"}
        
        # Try to load models
        self._load_models()

    def _load_models(self) -> None:
        """Load ML models - try pickle first, then fall back to .h5"""
        self.logger.info("Attempting to load audio ML models...")
        self.logger.info(f"  Current working directory: {os.getcwd()}")
        
        # First, try to load pickle-based models (new format)
        pickle_loaded = self._try_load_pickle_models()
        
        if pickle_loaded:
            self.logger.info("✅ Audio ML model ready (pickle format) - will use ML-based analysis")
            return
        
        # Fall back to .h5 model (legacy format)
        if self.model_path and os.path.exists(self.model_path):
            self.logger.info(f"Trying legacy .h5 model at: {self.model_path}")
            self._model = self._try_load_h5_model(self.model_path)
            if self._model is not None:
                self.logger.info("✅ Audio ML model ready (.h5 format) - will use ML-based analysis")
                return
        
        self.logger.warning("⚠️ Audio ML model not loaded - will use RULE-BASED analysis")

    def _try_load_pickle_models(self) -> bool:
        """Try to load pickle-based model files"""
        self.logger.info("Checking for pickle-based audio model files...")
        
        # Check weights file
        if not os.path.exists(self.weights_path):
            self.logger.warning(f"  Weights file not found: {os.path.abspath(self.weights_path)}")
            return False
        
        # Check scaler file
        if not os.path.exists(self.scaler_path):
            self.logger.warning(f"  Scaler file not found: {os.path.abspath(self.scaler_path)}")
            return False
        
        # Check label encoder file
        if not os.path.exists(self.label_encoder_path):
            self.logger.warning(f"  Label encoder file not found: {os.path.abspath(self.label_encoder_path)}")
            return False
        
        try:
            # Load weights
            self.logger.info(f"  Loading weights from: {self.weights_path}")
            with open(self.weights_path, "rb") as f:
                self._weights = pickle.load(f)
            self.logger.info(f"  ✅ Loaded weights")
            
            # Load scaler
            self.logger.info(f"  Loading scaler from: {self.scaler_path}")
            with open(self.scaler_path, "rb") as f:
                self._scaler = pickle.load(f)
            self.logger.info(f"  ✅ Loaded scaler")
            
            # Load label encoder
            self.logger.info(f"  Loading label encoder from: {self.label_encoder_path}")
            with open(self.label_encoder_path, "rb") as f:
                self._label_encoder = pickle.load(f)
            self.logger.info(f"  ✅ Loaded label encoder")
            if hasattr(self._label_encoder, 'classes_'):
                self.logger.info(f"     Classes: {list(self._label_encoder.classes_)}")
            
            # Optionally load architecture
            if os.path.exists(self.architecture_path):
                self.logger.info(f"  Loading architecture from: {self.architecture_path}")
                with open(self.architecture_path, "r") as f:
                    self._architecture = json.load(f)
                self.logger.info(f"  ✅ Loaded architecture")
            
            # Build model from weights if we have TensorFlow
            if HAS_TF and self._architecture:
                try:
                    self._model = self._build_model_from_weights()
                    if self._model:
                        self.logger.info("  ✅ Built TensorFlow model from weights")
                except Exception as e:
                    self.logger.warning(f"  Could not build TF model: {e}")
                    self.logger.info("  Will use numpy-based inference instead")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load pickle models: {e}")
            self._weights = None
            self._scaler = None
            self._label_encoder = None
            return False

    def _build_model_from_weights(self):
        """Build a TensorFlow model from architecture and weights"""
        if not HAS_TF or not self._architecture or not self._weights:
            return None
        
        try:
            # Try to build model from JSON config
            model = tf.keras.models.model_from_json(json.dumps(self._architecture))
            
            # Set weights
            if isinstance(self._weights, list):
                model.set_weights(self._weights)
            
            return model
        except Exception as e:
            self.logger.warning(f"Could not build model from architecture: {e}")
            return None

    def _try_load_h5_model(self, path: str):
        """Try to load legacy .h5 TensorFlow model"""
        if not HAS_TF:
            self.logger.warning("TensorFlow not available for .h5 model loading")
            return None
            
        try:
            return tf.keras.models.load_model(path, compile=False)
        except Exception as e1:
            self.logger.warning(f"Audio model load_model compile=False failed: {e1}")
            try:
                import h5py
                with h5py.File(path, "r") as f:
                    cfg_bytes = f.attrs.get("model_config", None)
                    if cfg_bytes is None and "model_config" in f:
                        cfg_bytes = f["model_config"][()]
                    if cfg_bytes is None:
                        raise RuntimeError("No model_config found in H5")
                    if isinstance(cfg_bytes, (bytes, bytearray)):
                        cfg_json = cfg_bytes.decode("utf-8")
                    else:
                        cfg_json = cfg_bytes
                cfg_json = cfg_json.replace('"batch_shape"', '"batch_input_shape"')
                model = tf.keras.models.model_from_json(cfg_json)
                try:
                    model.load_weights(path)
                except Exception as e_w:
                    self.logger.warning(f"Audio model load_weights warning: {e_w}")
                self.logger.info("Audio model loaded via JSON shim")
                return model
            except Exception as e2:
                self.logger.error(f"Audio model fallback failed: {e2}")
                return None

    def _record(self, out_wav: str) -> bool:
        """Extract audio from latest video clip (already has good audio)"""
        # Find latest video clip
        clips_dir = os.path.join(os.path.dirname(self.audio_dir), "clips")
        
        try:
            if os.path.exists(clips_dir):
                videos = sorted([f for f in os.listdir(clips_dir) if f.endswith('.mp4')], reverse=True)
                if videos:
                    latest_video = os.path.join(clips_dir, videos[0])
                    
                    # Extract audio from video (already boosted 60x)
                    cmd = [
                        "ffmpeg", "-i", latest_video,
                        "-vn", "-acodec", "pcm_s32le", "-ar", str(self.sample_rate),
                        "-ac", str(self.channels), "-y", out_wav
                    ]
                    result = subprocess.run(cmd, capture_output=True, timeout=30)
                    
                    if result.returncode == 0 and os.path.exists(out_wav):
                        self.logger.info(f"Extracted audio from {videos[0]}")
                        return True
            
            # Fallback: record directly with boost
            self.logger.info("No video found, recording directly")
            temp_wav = out_wav + ".tmp.wav"
            cmd = [
                "arecord",
                "-D", self.device,
                "-f", self.sample_format,
                "-r", str(self.sample_rate),
                "-c", str(self.channels),
                "-d", str(self.rec_duration),
                temp_wav
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Boost 60x
            subprocess.run([
                "ffmpeg", "-i", temp_wav, "-af", "volume=60.0", "-y", out_wav
            ], capture_output=True, timeout=30)
            
            try:
                os.remove(temp_wav)
            except:
                pass
            
            return os.path.exists(out_wav)
            
        except Exception as e:
            self.logger.error(f"Audio recording/extraction failed: {e}")
            return False

    def _extract_features(self, wav_path: str) -> np.ndarray:
        """Extract features for TensorFlow model (mel spectrogram)"""
        y, sr = librosa.load(wav_path, sr=self.sample_rate, mono=True)
        mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr//2)
        logmels = librosa.power_to_db(mels, ref=np.max)
        feat = logmels.T
        target_frames = 400
        if feat.shape[0] < target_frames:
            pad = np.zeros((target_frames - feat.shape[0], feat.shape[1]), dtype=feat.dtype)
            feat = np.vstack([feat, pad])
        else:
            feat = feat[:target_frames, :]
        return feat[np.newaxis, ..., np.newaxis].astype("float32")

    def _extract_features_for_pickle_model(self, wav_path: str) -> np.ndarray:
        """
        Extract features for pickle-based model
        Must produce exactly 157 features to match training
        
        Feature breakdown (157 total):
        - 13 MFCCs × 4 stats (mean, std, min, max) = 52
        - 13 MFCC deltas × 4 stats = 52  
        - 13 MFCC delta-deltas × 4 stats = 52
        - Spectral centroid (mean) = 1
        Total = 157
        """
        y, sr = librosa.load(wav_path, sr=self.sample_rate, mono=True, duration=30)
        
        features = []
        
        # MFCCs (13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # MFCC deltas (first derivative)
        mfcc_delta = librosa.feature.delta(mfccs)
        
        # MFCC delta-deltas (second derivative)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        
        # For each MFCC coefficient, compute 4 statistics
        for i in range(13):
            features.append(np.mean(mfccs[i]))
            features.append(np.std(mfccs[i]))
            features.append(np.min(mfccs[i]))
            features.append(np.max(mfccs[i]))
        # 52 features so far
        
        # For each MFCC delta coefficient, compute 4 statistics
        for i in range(13):
            features.append(np.mean(mfcc_delta[i]))
            features.append(np.std(mfcc_delta[i]))
            features.append(np.min(mfcc_delta[i]))
            features.append(np.max(mfcc_delta[i]))
        # 104 features so far
        
        # For each MFCC delta-delta coefficient, compute 4 statistics
        for i in range(13):
            features.append(np.mean(mfcc_delta2[i]))
            features.append(np.std(mfcc_delta2[i]))
            features.append(np.min(mfcc_delta2[i]))
            features.append(np.max(mfcc_delta2[i]))
        # 156 features so far
        
        # Add spectral centroid mean to get to 157
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features.append(np.mean(spectral_centroids))
        # 157 features total
        
        self.logger.debug(f"Extracted {len(features)} audio features")
        
        return np.array(features).reshape(1, -1)

    def _predict_with_pickle_model(self, wav_path: str) -> Tuple[str, float]:
        """Make prediction using pickle-based model"""
        try:
            # Extract features
            features = self._extract_features_for_pickle_model(wav_path)
            
            # Scale features if scaler is available
            if self._scaler is not None:
                features = self._scaler.transform(features)
            
            # If we have a TF model built from weights, use it
            if self._model is not None:
                probs = self._model.predict(features, verbose=0)
                if probs.ndim == 2 and probs.shape[0] == 1:
                    probs = probs[0]
                label_idx = int(np.argmax(probs))
                confidence = float(probs[label_idx])
            
            # Otherwise, do simple inference with weights (neural network forward pass)
            elif self._weights is not None:
                # Simple feedforward neural network inference
                x = features.flatten()
                
                # Forward pass through layers
                for i in range(0, len(self._weights) - 1, 2):
                    W = self._weights[i]
                    b = self._weights[i + 1]
                    x = np.dot(x, W) + b
                    # ReLU activation for hidden layers
                    if i < len(self._weights) - 2:
                        x = np.maximum(0, x)
                
                # Softmax for output
                exp_x = np.exp(x - np.max(x))
                probs = exp_x / exp_x.sum()
                
                label_idx = int(np.argmax(probs))
                confidence = float(probs[label_idx])
            else:
                raise RuntimeError("No model weights available")
            
            # Get label from encoder
            if self._label_encoder is not None:
                label = self._label_encoder.inverse_transform([label_idx])[0]
            else:
                label = self._labels.get(label_idx, str(label_idx))
            
            self.logger.info(f"ML audio analysis (pickle): {label} ({confidence:.2f})")
            return label, confidence
            
        except Exception as e:
            self.logger.error(f"Pickle model prediction failed: {e}")
            raise

    def _analyze_simple(self, wav_path: str) -> Tuple[str, float]:
        """
        Simple rule-based audio analysis (no ML model needed)
        Based on acoustic features of healthy vs unhealthy bee colonies
        """
        try:
            # Load audio
            y, sr = librosa.load(wav_path, sr=self.sample_rate, duration=30)
            
            # Calculate acoustic features
            # Healthy bee colonies have consistent frequency patterns around 200-400Hz
            # Unhealthy colonies show more variation and higher frequencies
            
            # 1. Spectral centroid (frequency center of mass)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            mean_centroid = np.mean(spectral_centroids)
            std_centroid = np.std(spectral_centroids)
            
            # 2. Zero crossing rate (noise indicator)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            mean_zcr = np.mean(zcr)
            
            # 3. RMS energy (volume/activity level)
            rms = librosa.feature.rms(y=y)[0]
            mean_rms = np.mean(rms)
            std_rms = np.std(rms)
            
            # 4. Spectral rolloff (frequency distribution)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            mean_rolloff = np.mean(rolloff)
            
            # Simple rule-based classification
            health_score = 0.5  # Start neutral
            
            # Good indicators (healthy colony)
            if 200 < mean_centroid < 800:  # Normal bee frequency range
                health_score += 0.2
            if 0.01 < mean_rms < 0.1:  # Normal activity level
                health_score += 0.15
            if mean_zcr < 0.15:  # Low noise/chaos
                health_score += 0.15
            if std_centroid < 200:  # Consistent frequency
                health_score += 0.1
            
            # Bad indicators (unhealthy/stressed colony)
            if mean_centroid > 1000:  # Very high pitched (stress)
                health_score -= 0.3
            if mean_centroid < 100:  # Very low (weak/dying)
                health_score -= 0.3
            if mean_rms < 0.005:  # Too quiet (inactive)
                health_score -= 0.25
            if mean_rms > 0.15:  # Too loud (agitated)
                health_score -= 0.2
            if mean_zcr > 0.2:  # High noise/chaos
                health_score -= 0.2
            if std_centroid > 300:  # Erratic frequency
                health_score -= 0.15
            
            # Clamp score
            health_score = max(0.0, min(1.0, health_score))
            
            # Classify
            if health_score >= 0.6:
                label = "healthy"
                confidence = health_score
            else:
                label = "unhealthy"
                confidence = 1.0 - health_score
            
            self.logger.info(
                f"Simple audio analysis: centroid={mean_centroid:.1f}Hz, "
                f"rms={mean_rms:.4f}, zcr={mean_zcr:.4f} -> {label} ({confidence:.2f})"
            )
            
            return label, confidence
            
        except Exception as e:
            self.logger.error(f"Simple audio analysis error: {e}")
            return "unknown", 0.0

    def analyze(self, wav_path: str) -> Tuple[str, float]:
        """
        Analyze audio file for bee health
        Uses ML model if available, otherwise falls back to rule-based analysis
        """
        self.logger.info(f"Analyzing audio: {wav_path}")
        
        # Try pickle-based model first (new format)
        if self._weights is not None or (self._model is not None and self._scaler is not None):
            try:
                return self._predict_with_pickle_model(wav_path)
            except Exception as e:
                self.logger.warning(f"Pickle model analysis failed: {e}, trying other methods")
        
        # Try TF model if available (legacy .h5 format)
        if self._model is not None and self._scaler is None:
            try:
                x = self._extract_features(wav_path)
                probs = self._model.predict(x, verbose=0)
                if probs.ndim == 2 and probs.shape[0] == 1:
                    probs = probs[0]
                label_idx = int(np.argmax(probs))
                confidence = float(probs[label_idx])
                label = self._labels.get(label_idx, str(label_idx))
                self.logger.info(f"ML audio analysis (.h5): {label} ({confidence:.2f})")
                return label, confidence
            except Exception as e:
                self.logger.warning(f"TF model analysis failed: {e}, using fallback")
        
        # Fall back to simple rule-based analysis
        self.logger.info("Using RULE-BASED audio analysis")
        return self._analyze_simple(wav_path)
    
    def is_using_ml_model(self) -> bool:
        """Check if ML model is loaded and will be used"""
        return (self._weights is not None) or (self._model is not None)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model"""
        return {
            "ml_model_loaded": self.is_using_ml_model(),
            "model_type": "pickle" if self._weights is not None else ("h5" if self._model is not None else "none"),
            "has_scaler": self._scaler is not None,
            "has_label_encoder": self._label_encoder is not None,
            "labels": list(self._label_encoder.classes_) if self._label_encoder and hasattr(self._label_encoder, 'classes_') else list(self._labels.values())
        }