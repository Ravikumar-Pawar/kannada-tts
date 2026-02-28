"""
Hybrid Models - VITS (Variational Inference TTS) for superior quality
"""

from .vits_model import VITS
from .tacotron2_hybrid import Tacotron2Hybrid
from .vocoder_hybrid import VocoderHybrid

__all__ = ["VITS", "Tacotron2Hybrid", "VocoderHybrid"]
