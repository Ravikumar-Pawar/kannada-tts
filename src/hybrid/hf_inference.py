"""
Inference wrapper for HuggingFace MMS-TTS Kannada VITS model
"""

import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


class HFVITSInference:
    """Wraps a HuggingFace VitsModel for Kannada TTS."""

    def __init__(self, hf_model, tokenizer,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = hf_model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        # sampling rate defined in config
        self.sample_rate = getattr(self.model.config, "sampling_rate", 22050)
        logger.info(f"HFVITSInference initialized on {device} (sr={self.sample_rate})")

    def synthesize(self, text: str, **kwargs) -> np.ndarray:
        """Generate audio from input text using HF VITS.
        Other kwargs are ignored; the HuggingFace model doesn't use emotions.
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**inputs).waveform
        # waveform: (batch, length) or (batch, channels, length?)
        audio = output.squeeze().cpu().numpy()
        return audio

    def save_audio(self, audio: np.ndarray, output_path: str):
        import soundfile as sf
        sf.write(output_path, audio, self.sample_rate)
        logger.info(f"Audio saved to {output_path}")

    def get_info(self):
        return {
            "model_type": "hf_vits",
            "device": str(self.device),
            "sample_rate": self.sample_rate
        }
