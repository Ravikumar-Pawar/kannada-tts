"""
Unified Inference Interface
Supports both hybrid and non-hybrid approaches
"""

import torch
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TTSInference:
    """Unified TTS Inference Interface"""
    
    def __init__(self, approach: str = "hybrid", 
                 tacotron2_model=None,
                 vocoder_model=None,
                 character_mapping: Optional[dict] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize unified inference
        
        Args:
            approach: "hybrid" or "non_hybrid"
            tacotron2_model: Acoustic model
            vocoder_model: Vocoder model
            character_mapping: Character mapping
            device: Device to use
        """
        self.approach = approach.lower()
        self.device = device
        
        if self.approach == "hybrid":
            # "Hybrid" in the web UI refers to VITS; could be HF pretrained
            # object returned from ModelManager is a dict when hf model.
            if isinstance(tacotron2_model, dict) and tacotron2_model.get('hf'):
                from src.hybrid.hf_inference import HFVITSInference
                self.inference = HFVITSInference(
                    tacotron2_model['model'], tacotron2_model['tokenizer'], device
                )
                logger.info("Using HuggingFace VITS (hybrid) inference")
            else:
                from src.hybrid.vits_inference import VITSInference
                self.inference = VITSInference(
                    tacotron2_model, character_mapping, device
                )
                logger.info("Using VITS (hybrid) approach for inference")
        elif self.approach == "non_hybrid":
            from src.non_hybrid.inference import StandardInference
            self.inference = StandardInference(
                tacotron2_model, vocoder_model, character_mapping, device
            )
            logger.info("Using NON-HYBRID approach for inference")
        else:
            raise ValueError(f"Unknown approach: {approach}")
    
    def synthesize(self, text: str, **kwargs):
        """
        Synthesize speech from text
        
        Args:
            text: Input text
            **kwargs: Approach-specific parameters
        
        Returns:
            Audio waveform
        """
        if self.approach == "hybrid":
            # VITSInference expects emotion/post_processing additionally
            return self.inference.synthesize(
                text,
                emotion=kwargs.get("emotion", "neutral"),
                post_processing=kwargs.get("post_processing", "advanced"),
                **{k: v for k, v in kwargs.items()
                   if k not in ["emotion", "post_processing"]}
            )
        else:
            # drop any None values (e.g. hybrid-specific args passed as None)
            filtered = {k: v for k, v in kwargs.items() if v is not None}
            return self.inference.synthesize(text, **filtered)
    
    def synthesize_batch(self, texts: list, **kwargs):
        """Synthesize multiple texts"""
        if self.approach == "hybrid":
            return self.inference.synthesize_batch(
                texts,
                emotion=kwargs.get("emotion", "neutral"),
                post_processing=kwargs.get("post_processing", "advanced")
            )
        else:
            return self.inference.synthesize_batch(texts)
    
    def save_audio(self, audio: np.ndarray, output_path: str):
        """Save audio to file"""
        self.inference.save_audio(audio, output_path)
    
    def get_info(self):
        """Get inference information"""
        return {
            "approach": self.approach,
            "device": str(self.device),
            "sample_rate": self.inference.sample_rate
        }
