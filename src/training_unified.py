"""
Unified Training Interface
Supports both hybrid and non-hybrid approaches
"""

import torch
import torch.nn as nn
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class TTSTrainer:
    """Unified TTS Training Interface"""
    
    def __init__(self, approach: str = "hybrid",
                 tacotron2_model: nn.Module = None,
                 vocoder_model: nn.Module = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize unified trainer
        
        Args:
            approach: "hybrid" or "non_hybrid"
            tacotron2_model: Acoustic model
            vocoder_model: Vocoder model
            device: Device to use
        """
        self.approach = approach.lower()
        self.device = device
        
        if self.approach == "hybrid":
            from src.hybrid.training import HybridTrainer
            self.trainer = HybridTrainer(tacotron2_model, vocoder_model, device)
            logger.info("Using HYBRID approach for training")
        elif self.approach == "non_hybrid":
            from src.non_hybrid.training import StandardTrainer
            self.trainer = StandardTrainer(tacotron2_model, vocoder_model, device)
            logger.info("Using NON-HYBRID approach for training")
        else:
            raise ValueError(f"Unknown approach: {approach}")
    
    def train_epoch(self, train_loader, val_loader, epoch: int) -> Dict:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epoch: Epoch number
        
        Returns:
            Metrics dictionary
        """
        return self.trainer.train_epoch(train_loader, val_loader, epoch)
    
    def save_checkpoint(self, epoch: int, metrics: Dict):
        """Save training checkpoint"""
        self.trainer.save_checkpoint(epoch, metrics)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        return self.trainer.load_checkpoint(checkpoint_path)
    
    def get_info(self):
        """Get trainer information"""
        return {
            "approach": self.approach,
            "device": str(self.device),
            "checkpoint_dir": self.trainer.checkpoint_dir
        }
