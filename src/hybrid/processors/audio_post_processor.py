"""
Audio Post-Processor for Hybrid Approach
Combines multiple processors for optimal output quality
"""

import numpy as np
import logging
from typing import Optional
from .noise_reduction import NoiseReductionProcessor
from .prosody_enhancement import ProsodyEnhancer

logger = logging.getLogger(__name__)


class AudioPostProcessor:
    """Post-processor combining multiple audio enhancement techniques"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sr = sample_rate
        self.noise_reducer = NoiseReductionProcessor(sample_rate)
        self.prosody_enhancer = ProsodyEnhancer(sample_rate)
        logger.info(f"AudioPostProcessor initialized with sr={sample_rate}")
    
    def process_standard(self, y: np.ndarray) -> np.ndarray:
        """
        Standard post-processing pipeline
        
        Args:
            y: Audio signal
        
        Returns:
            Processed audio
        """
        # 1. Denoise with spectral gating
        y = self.noise_reducer.denoise(y, method="spectral_gating", normalize=False)
        
        # 2. Normalize loudness
        y = self.noise_reducer.normalize_loudness(y, target_db=-20.0)
        
        logger.info("Applied standard post-processing")
        return y
    
    def process_advanced(self, y: np.ndarray) -> np.ndarray:
        """
        Advanced post-processing pipeline with prosody enhancement
        
        Args:
            y: Audio signal
        
        Returns:
            Processed audio
        """
        # 1. Advanced multi-stage denoising
        y = self.noise_reducer.denoise_advanced(y, normalize=False)
        
        # 2. Prosody enhancement
        y = self.prosody_enhancer.enhance_prosody(
            y,
            pitch_shift=1.0,
            time_stretch=1.0,
            compress_dynamics=True,
            add_vibrato=False
        )
        
        # 3. Final normalization
        y = self.noise_reducer.normalize_loudness(y, target_db=-20.0)
        
        logger.info("Applied advanced post-processing")
        return y
    
    def process_quality_focus(self, y: np.ndarray) -> np.ndarray:
        """
        Quality-focused post-processing for max audio quality
        
        Args:
            y: Audio signal
        
        Returns:
            Processed audio
        """
        # 1. Wiener filtering
        y = self.noise_reducer.wiener_filter(y, noise_profile_duration=0.5)
        
        # 2. Median filtering
        y = self.noise_reducer.median_filter(y, kernel_size=5)
        
        # 3. Dynamic enhancement
        y = self.prosody_enhancer.enhance_dynamics(y, compression_ratio=4.0)
        
        # 4. Add subtle vibrato
        y = self.prosody_enhancer.add_vibrato(y, speed=5.0, depth=0.02)
        
        # 5. Final normalization
        y = self.noise_reducer.normalize_loudness(y, target_db=-20.0)
        
        logger.info("Applied quality-focused post-processing")
        return y
    
    def process_speed_focus(self, y: np.ndarray) -> np.ndarray:
        """
        Speed-focused post-processing for fast inference
        
        Args:
            y: Audio signal
        
        Returns:
            Processed audio
        """
        # 1. Quick spectral gating
        y = self.noise_reducer.spectral_gating(y, threshold_db=-40)
        
        # 2. Quick normalization
        y = self.noise_reducer.normalize_loudness(y, target_db=-20.0)
        
        logger.info("Applied speed-focused post-processing")
        return y
    
    def process_with_emotion(self, y: np.ndarray, emotion: str = "neutral") -> np.ndarray:
        """
        Post-processing with emotional control
        
        Args:
            y: Audio signal
            emotion: Emotional tone
        
        Returns:
            Processed audio with emotion
        """
        # 1. Denoise
        y = self.noise_reducer.denoise(y, method="spectral_gating", normalize=False)
        
        # 2. Apply emotion
        y = self.prosody_enhancer.emotion_control(y, emotion=emotion)
        
        # 3. Normalize
        y = self.noise_reducer.normalize_loudness(y, target_db=-20.0)
        
        logger.info(f"Applied post-processing with {emotion} emotion")
        return y
    
    def process(self, y: np.ndarray, 
                pipeline: str = "standard",
                emotion: Optional[str] = None,
                **kwargs) -> np.ndarray:
        """
        Apply post-processing with various pipelines
        
        Args:
            y: Audio signal
            pipeline: "standard", "advanced", "quality", "speed"
            emotion: Optional emotion for control
            **kwargs: Additional parameters
        
        Returns:
            Processed audio
        """
        if emotion:
            return self.process_with_emotion(y, emotion)
        
        if pipeline == "standard":
            return self.process_standard(y)
        elif pipeline == "advanced":
            return self.process_advanced(y)
        elif pipeline == "quality":
            return self.process_quality_focus(y)
        elif pipeline == "speed":
            return self.process_speed_focus(y)
        else:
            logger.warning(f"Unknown pipeline: {pipeline}, using standard")
            return self.process_standard(y)
    
    def batch_process(self, audio_list: list, 
                     pipeline: str = "standard",
                     emotion: Optional[str] = None) -> list:
        """
        Process multiple audio samples
        
        Args:
            audio_list: List of audio arrays
            pipeline: Processing pipeline
            emotion: Optional emotion
        
        Returns:
            List of processed audio arrays
        """
        processed = []
        for idx, audio in enumerate(audio_list):
            try:
                processed_audio = self.process(audio, pipeline=pipeline, emotion=emotion)
                processed.append(processed_audio)
            except Exception as e:
                logger.error(f"Error processing audio {idx}: {e}")
                processed.append(audio)  # Return original on error
        
        return processed
