"""
Hybrid TTS Package - Advanced Tacotron2 + HiFiGAN with Post-Processing
Combines standard models with advanced audio processing techniques
"""

from . import models
from . import processors
from .inference import HybridInference
from .training import HybridTrainer

__version__ = "1.0.0"
__all__ = ["models", "processors", "HybridInference", "HybridTrainer"]
