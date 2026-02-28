"""
Non-Hybrid TTS Package - Standard Tacotron2 + HiFiGAN Approach
Basic configuration without advanced features
"""

from . import models
from .inference import StandardInference
from .training import StandardTrainer

__version__ = "1.0.0"
__all__ = ["models", "StandardInference", "StandardTrainer"]
