"""
Noise Reduction Processor for Hybrid Approach
Advanced spectral gating, Wiener filtering, and denoising
"""

import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class NoiseReductionProcessor:
    """Advanced noise reduction for TTS output"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sr = sample_rate
        logger.info(f"NoiseReductionProcessor initialized with sr={sample_rate}")
    
    def spectral_gating(self, y: np.ndarray, threshold_db: float = -40, 
                       fft_size: int = 2048) -> np.ndarray:
        """
        Spectral gating: suppress frequencies below threshold
        
        Args:
            y: Audio signal
            threshold_db: Threshold in dB
            fft_size: FFT size for STFT
        
        Returns:
            Denoised audio
        """
        try:
            import librosa
        except ImportError:
            logger.warning("librosa not available, returning original audio")
            return y
        
        D = librosa.stft(y, n_fft=fft_size)
        magnitude = np.abs(D)
        
        # Compute power spectrum
        db = librosa.power_to_db(magnitude**2, ref=np.max)
        threshold = np.percentile(db, 15)  # 15th percentile
        
        # Create mask
        mask = db > threshold
        D_masked = D * mask
        
        # Reconstruct
        y_denoised = librosa.istft(D_masked)
        
        logger.info(f"Applied spectral gating (threshold={threshold_db}dB)")
        return y_denoised
    
    def wiener_filter(self, y: np.ndarray, noise_profile_duration: float = 0.5) -> np.ndarray:
        """
        Wiener filtering for noise reduction
        
        Args:
            y: Audio signal
            noise_profile_duration: Duration of noise profile in seconds
        
        Returns:
            Filtered audio
        """
        try:
            import librosa
        except ImportError:
            logger.warning("librosa not available, returning original audio")
            return y
        
        # Estimate noise from first part
        noise_len = int(self.sr * noise_profile_duration)
        noise_len = min(noise_len, len(y) // 4)
        
        if noise_len < 512:
            return y
        
        noise = y[:noise_len]
        
        # Compute noise power spectrum
        D_noise = librosa.stft(noise)
        noise_power = np.abs(D_noise) ** 2
        
        D = librosa.stft(y)
        magnitude = np.abs(D)
        
        # Wiener filter coefficient
        signal_power = magnitude ** 2
        wiener_coeff = signal_power / (signal_power + np.mean(noise_power, axis=1, keepdims=True) + 1e-8)
        wiener_coeff = np.maximum(wiener_coeff, 0.1)  # Floor at 0.1
        
        D_filtered = D * wiener_coeff
        y_denoised = librosa.istft(D_filtered)
        
        logger.info(f"Applied Wiener filtering")
        return y_denoised
    
    def median_filter(self, y: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Apply median filtering to reduce impulsive noise
        
        Args:
            y: Audio signal
            kernel_size: Median filter kernel size
        
        Returns:
            Filtered audio
        """
        try:
            from scipy.signal import medfilt
        except ImportError:
            logger.warning("scipy not available, returning original audio")
            return y
        
        y_filtered = medfilt(y, kernel_size=kernel_size)
        logger.info(f"Applied median filtering (kernel={kernel_size})")
        return y_filtered
    
    def normalize_loudness(self, y: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """
        Normalize audio loudness
        
        Args:
            y: Audio signal
            target_db: Target loudness in dB
        
        Returns:
            Normalized audio
        """
        try:
            import librosa
        except ImportError:
            logger.warning("librosa not available, returning original audio")
            return y
        
        # Calculate current loudness
        current_db = librosa.feature.melspectrogram(y=y, sr=self.sr)
        current_loudness = np.mean(librosa.power_to_db(current_db))
        
        # Calculate gain
        gain_db = target_db - current_loudness
        gain_linear = 10 ** (gain_db / 20)
        
        # Apply gain
        y_normalized = y * gain_linear
        
        # Prevent clipping
        y_normalized = np.clip(y_normalized, -1.0, 1.0)
        
        logger.info(f"Normalized loudness (gain={gain_db:.2f}dB)")
        return y_normalized
    
    def denoise(self, y: np.ndarray, 
                method: str = "spectral_gating",
                normalize: bool = True) -> np.ndarray:
        """
        Apply noise reduction with optional normalization
        
        Args:
            y: Audio signal
            method: "spectral_gating", "wiener", or "median"
            normalize: Whether to normalize loudness
        
        Returns:
            Denoised audio
        """
        if method == "spectral_gating":
            y_denoised = self.spectral_gating(y)
        elif method == "wiener":
            y_denoised = self.wiener_filter(y)
        elif method == "median":
            y_denoised = self.median_filter(y)
        else:
            logger.warning(f"Unknown denoising method: {method}, returning original")
            y_denoised = y
        
        if normalize:
            y_denoised = self.normalize_loudness(y_denoised)
        
        return y_denoised
    
    def denoise_advanced(self, y: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Apply advanced multi-stage denoising
        
        Args:
            y: Audio signal
            normalize: Whether to normalize loudness
        
        Returns:
            Denoised audio
        """
        # Stage 1: Spectral gating
        y = self.spectral_gating(y)
        
        # Stage 2: Median filter
        y = self.median_filter(y, kernel_size=3)
        
        # Stage 3: Wiener filter
        y = self.wiener_filter(y)
        
        # Stage 4: Normalization
        if normalize:
            y = self.normalize_loudness(y)
        
        logger.info("Applied advanced multi-stage denoising")
        return y
