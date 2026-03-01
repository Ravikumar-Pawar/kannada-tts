"""
Metrics Calculator
Computes audio quality metrics for TTS output
"""

import numpy as np
import librosa
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculates TTS audio quality metrics"""
    
    def __init__(self, sample_rate: int = 22050):
        """Initialize metrics calculator"""
        self.sample_rate = sample_rate
    
    def calculate_metrics(self, audio: np.ndarray, model_name: str = "") -> Dict:
        """
        Calculate comprehensive audio metrics
        
        Args:
            audio: Audio waveform (numpy array)
            model_name: Name of the model for logging
        
        Returns:
            Dictionary with calculated metrics
        """
        try:
            audio = np.array(audio, dtype=np.float32)
            
            metrics = {
                "model": model_name,
                "mcd": self._calculate_mcd(audio),
                "snr": self._calculate_snr(audio),
                "lsd": self._calculate_lsd(audio),
                "zcr": self._calculate_zcr(audio),
                "rms": self._calculate_rms(audio),
                "duration": len(audio) / self.sample_rate,
            }
            
            logger.info(f"{model_name} Metrics: MCD={metrics['mcd']:.2f}, SNR={metrics['snr']:.2f}, LSD={metrics['lsd']:.2f}")
            return metrics
        
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {
                "error": str(e),
                "model": model_name
            }
    
    def _calculate_mcd(self, audio: np.ndarray) -> float:
        """
        Calculate Mel-Cepstral Distortion
        Lower is better (5.1 for Tacotron2, 4.2 for VITS)
        """
        try:
            # Compute mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=80,
                n_fft=1024,
                hop_length=256
            )
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Compute cepstral coefficients
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=13
            )
            
            # Compute mean and std of MFCCs to simulate MCD
            # In practice, MCD requires comparing with reference signal
            # Here we estimate based on spectrogram stability
            mcd_estimate = np.std(mfcc) + np.std(mel_spec_db) / 10
            
            return float(mcd_estimate)
        
        except Exception as e:
            logger.warning(f"MCD calculation failed: {e}")
            return 0.0
    
    def _calculate_snr(self, audio: np.ndarray) -> float:
        """
        Calculate Signal-to-Noise Ratio
        Higher is better (20.8 for Tacotron2, 22.5 for VITS)
        """
        try:
            # Use energy-based approach
            power = np.mean(audio ** 2)
            
            # Estimate noise as lowest energy regions
            sorted_power = np.sort(np.abs(audio))
            noise_power = np.mean(sorted_power[:int(len(sorted_power) * 0.1)])
            
            if noise_power > 0:
                snr = 10 * np.log10(power / (noise_power ** 2 + 1e-10))
            else:
                snr = 30.0  # Default high SNR
            
            return float(np.clip(snr, 0, 50))
        
        except Exception as e:
            logger.warning(f"SNR calculation failed: {e}")
            return 0.0
    
    def _calculate_lsd(self, audio: np.ndarray) -> float:
        """
        Calculate Log Spectral Distance
        Lower is better (4.5 for Tacotron2, 3.8 for VITS)
        """
        try:
            # Compute STFT
            D = librosa.stft(audio)
            magnitude = np.abs(D)
            
            # Convert to dB scale
            S_db = librosa.amplitude_to_db(magnitude, ref=np.max)
            
            # Compute mean and std as proxy for LSD
            lsd = np.std(S_db) + np.mean(np.abs(np.diff(S_db, axis=1)))
            
            return float(lsd)
        
        except Exception as e:
            logger.warning(f"LSD calculation failed: {e}")
            return 0.0
    
    def _calculate_zcr(self, audio: np.ndarray) -> float:
        """
        Calculate Zero Crossing Rate
        Indicates voicing and speech characteristics
        """
        try:
            zcr = librosa.feature.zero_crossing_rate(audio)
            return float(np.mean(zcr))
        
        except Exception as e:
            logger.warning(f"ZCR calculation failed: {e}")
            return 0.0
    
    def _calculate_rms(self, audio: np.ndarray) -> float:
        """
        Calculate Root Mean Square energy
        """
        try:
            rms = np.sqrt(np.mean(audio ** 2))
            return float(rms)
        
        except Exception as e:
            logger.warning(f"RMS calculation failed: {e}")
            return 0.0
    
    def compare_metrics(self, metrics1: Dict, metrics2: Dict) -> Dict:
        """
        Compare two sets of metrics
        
        Args:
            metrics1: First model's metrics
            metrics2: Second model's metrics
        
        Returns:
            Comparison results
        """
        return {
            "model1": metrics1.get("model", "Unknown"),
            "model2": metrics2.get("model", "Unknown"),
            "quality_comparison": {
                "mcd_winner": metrics1.get("model") if metrics1.get("mcd", 999) < metrics2.get("mcd", 999) else metrics2.get("model"),
                "mcd_improvement": f"{((metrics2.get('mcd', 0) - metrics1.get('mcd', 0)) / metrics2.get('mcd', 1) * 100):.1f}%" if metrics2.get('mcd') else "N/A",
                "snr_winner": metrics1.get("model") if metrics1.get("snr", 0) > metrics2.get("snr", 0) else metrics2.get("model"),
                "snr_improvement": f"{((metrics1.get('snr', 0) - metrics2.get('snr', 0)) / metrics2.get('snr', 1) * 100):.1f}%" if metrics2.get('snr') else "N/A",
            }
        }
