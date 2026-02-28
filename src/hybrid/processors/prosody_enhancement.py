"""
Prosody Enhancement Processor for Hybrid Approach
Enhance pitch, duration, and emotional expression
"""

import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class ProsodyEnhancer:
    """Enhance prosody aspects of TTS output"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sr = sample_rate
        logger.info(f"ProsodyEnhancer initialized with sr={sample_rate}")
    
    def pitch_shift(self, y: np.ndarray, shift_factor: float = 1.0) -> np.ndarray:
        """
        Shift pitch of audio signal
        
        Args:
            y: Audio signal
            shift_factor: Pitch shift factor (< 1.0 lowers, > 1.0 raises)
        
        Returns:
            Pitch-shifted audio
        """
        if shift_factor == 1.0:
            return y
        
        try:
            import librosa
        except ImportError:
            logger.warning("librosa not available, returning original audio")
            return y
        
        # Extract STFT
        D = librosa.stft(y)
        
        # Shift frequencies
        if shift_factor > 1.0:
            # Raise pitch by shifting frequencies up
            D_shifted = np.zeros_like(D)
            shift_bins = int(len(D) * (shift_factor - 1))
            D_shifted[shift_bins:] = D[:-shift_bins]
        else:
            # Lower pitch by shifting frequencies down
            D_shifted = np.zeros_like(D)
            drop_bins = int(len(D) * (1.0 - shift_factor))
            D_shifted[:-drop_bins] = D[drop_bins:]
        
        # Reconstruct
        y_shifted = librosa.istft(D_shifted)
        
        logger.info(f"Applied pitch shift (factor={shift_factor})")
        return y_shifted
    
    def time_stretch(self, y: np.ndarray, stretch_factor: float = 1.0) -> np.ndarray:
        """
        Time-stretch audio without changing pitch
        
        Args:
            y: Audio signal
            stretch_factor: Time stretch factor (< 1.0 faster, > 1.0 slower)
        
        Returns:
            Time-stretched audio
        """
        if stretch_factor == 1.0:
            return y
        
        try:
            import librosa
        except ImportError:
            logger.warning("librosa not available, returning original audio")
            return y
        
        D = librosa.stft(y)
        D_stretched = librosa.phase_vocoder(D, stretch_factor)
        y_stretched = librosa.istft(D_stretched)
        
        logger.info(f"Applied time stretch (factor={stretch_factor})")
        return y_stretched
    
    def enhance_dynamics(self, y: np.ndarray, compression_ratio: float = 4.0) -> np.ndarray:
        """
        Enhance dynamic range through dynamic range compression
        
        Args:
            y: Audio signal
            compression_ratio: Compression ratio (higher = more compression)
        
        Returns:
            Compressed audio
        """
        # Calculate envelope
        abs_signal = np.abs(y)
        
        # Smooth the signal
        window_size = int(0.05 * self.sr)  # 50ms window
        if window_size % 2 == 0:
            window_size += 1
        
        try:
            from scipy.signal import savgol_filter
            envelope = savgol_filter(abs_signal, window_size, 3)
        except:
            # Fallback: simple moving average
            kernel = np.ones(window_size) / window_size
            envelope = np.convolve(abs_signal, kernel, mode='same')
        
        # Apply compression
        threshold = np.max(envelope) / compression_ratio
        compressed_env = np.where(
            envelope > threshold,
            threshold + (envelope - threshold) / compression_ratio,
            envelope
        )
        
        # Apply gain
        gain = np.where(envelope > 1e-6, compressed_env / envelope, 1.0)
        y_compressed = y * gain
        
        logger.info(f"Enhanced dynamics (compression_ratio={compression_ratio})")
        return y_compressed
    
    def add_vibrato(self, y: np.ndarray, speed: float = 5.0, depth: float = 0.05) -> np.ndarray:
        """
        Add vibrato effect to enhance naturalness
        
        Args:
            y: Audio signal
            speed: Vibrato speed in Hz (typical: 4-8)
            depth: Vibrato depth as fraction of original (typical: 0.05-0.1)
        
        Returns:
            Audio with vibrato
        """
        n_samples = len(y)
        
        # Generate vibrato LFO
        t = np.arange(n_samples) / self.sr
        lfo = depth * np.sin(2 * np.pi * speed * t)
        
        # Apply vibrato through phase modulation
        try:
            import librosa
            
            D = librosa.stft(y)
            phase = np.angle(D)
            magnitude = np.abs(D)
            
            # Frequency bins
            freqs = librosa.fft_frequencies(sr=self.sr, n_fft=D.shape[0] * 2 - 2)
            
            # Time-varying frequency modulation
            time_frames = D.shape[1]
            phase_mod = np.zeros_like(phase)
            for t_idx in range(time_frames):
                time_in_sec = t_idx * (len(y) / self.sr) / time_frames
                lfo_val = depth * np.sin(2 * np.pi * speed * time_in_sec)
                phase_mod[:, t_idx] = phase[:, t_idx] + lfo_val
            
            D_vibrato = magnitude * np.exp(1j * phase_mod)
            y_vibrato = librosa.istft(D_vibrato)
            
            logger.info(f"Added vibrato (speed={speed}Hz, depth={depth})")
            return y_vibrato
        except:
            logger.warning("Could not add vibrato, returning original audio")
            return y
    
    def enhance_prosody(self, y: np.ndarray,
                       pitch_shift: float = 1.0,
                       time_stretch: float = 1.0,
                       compress_dynamics: bool = True,
                       add_vibrato: bool = False) -> np.ndarray:
        """
        Comprehensive prosody enhancement
        
        Args:
            y: Audio signal
            pitch_shift: Pitch shift factor
            time_stretch: Time stretch factor
            compress_dynamics: Whether to compress dynamics
            add_vibrato: Whether to add vibrato
        
        Returns:
            Enhanced audio
        """
        # Time stretch (affects duration)
        if time_stretch != 1.0:
            y = self.time_stretch(y, time_stretch)
        
        # Pitch shift
        if pitch_shift != 1.0:
            y = self.pitch_shift(y, pitch_shift)
        
        # Compress dynamics
        if compress_dynamics:
            y = self.enhance_dynamics(y, compression_ratio=4.0)
        
        # Add vibrato for naturalness
        if add_vibrato:
            y = self.add_vibrato(y, speed=5.0, depth=0.03)
        
        logger.info("Applied prosody enhancement")
        return y
    
    def emotion_control(self, y: np.ndarray, emotion: str = "neutral") -> np.ndarray:
        """
        Apply emotional prosody control
        
        Args:
            y: Audio signal
            emotion: "neutral", "happy", "sad", "angry", "surprised"
        
        Returns:
            Emotionally modulated audio
        """
        emotion_mapping = {
            "neutral": {"pitch": 1.0, "speed": 1.0, "compression": 3.0},
            "happy": {"pitch": 1.1, "speed": 0.9, "compression": 2.0},
            "sad": {"pitch": 0.9, "speed": 1.2, "compression": 5.0},
            "angry": {"pitch": 1.2, "speed": 0.7, "compression": 2.5},
            "surprised": {"pitch": 1.15, "speed": 0.85, "compression": 3.5}
        }
        
        if emotion not in emotion_mapping:
            logger.warning(f"Unknown emotion: {emotion}, using neutral")
            emotion = "neutral"
        
        params = emotion_mapping[emotion]
        
        # Apply time stretch first (inverse of speed)
        y = self.time_stretch(y, 1.0 / params["speed"])
        
        # Pitch shift
        y = self.pitch_shift(y, params["pitch"])
        
        # Dynamic compression
        y = self.enhance_dynamics(y, compression_ratio=params["compression"])
        
        logger.info(f"Applied {emotion} emotion control")
        return y
