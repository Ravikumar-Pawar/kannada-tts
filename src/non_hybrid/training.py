"""
Standard Training Pipeline for Non-Hybrid Approach
"""

import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class StandardTrainer:
    """Standard Trainer for Tacotron2 + HiFiGAN"""
    
    def __init__(self, tacotron2_model: nn.Module, 
                 vocoder_model: nn.Module,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize trainer
        
        Args:
            tacotron2_model: Acoustic model
            vocoder_model: Vocoder model
            device: Device to use
        """
        self.tacotron2 = tacotron2_model.to(device)
        self.vocoder = vocoder_model.to(device)
        self.device = device
        
        # Optimizers
        self.tacotron2_optimizer = optim.Adam(self.tacotron2.parameters(), lr=1e-3)
        self.vocoder_optimizer = optim.Adam(self.vocoder.parameters(), lr=1e-4)
        
        # Loss functions
        self.mel_loss = nn.L1Loss()
        self.gate_loss = nn.BCEWithLogitsLoss()
        
        # Checkpoints directory
        self.checkpoint_dir = "output/non_hybrid_checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        logger.info(f"StandardTrainer initialized on {device}")
    
    def train_tacotron2_step(self, text_input: torch.Tensor,
                            text_lengths: torch.Tensor,
                            target_mels: torch.Tensor,
                            target_gates: torch.Tensor) -> Dict[str, float]:
        """
        Single training step for Tacotron2
        
        Args:
            text_input: Character indices
            text_lengths: Text lengths
            target_mels: Target mel spectrograms
            target_gates: Target gate values
        
        Returns:
            Loss dictionary
        """
        self.tacotron2.train()
        
        # Forward pass
        mel_outputs, gate_outputs, _ = self.tacotron2(text_input, text_lengths, target_mels)
        
        # Compute losses
        mel_loss = self.mel_loss(mel_outputs, target_mels)
        gate_loss = self.gate_loss(gate_outputs.squeeze(-1), target_gates)
        
        total_loss = mel_loss + gate_loss
        
        # Backward pass
        self.tacotron2_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.tacotron2.parameters(), max_norm=1.0)
        self.tacotron2_optimizer.step()
        
        return {
            "tacotron2_mel_loss": mel_loss.item(),
            "tacotron2_gate_loss": gate_loss.item(),
            "tacotron2_total_loss": total_loss.item()
        }
    
    def train_vocoder_step(self, target_mels: torch.Tensor,
                          target_audio: torch.Tensor) -> Dict[str, float]:
        """
        Single training step for vocoder
        
        Args:
            target_mels: Target mel spectrograms
            target_audio: Target audio waveforms
        
        Returns:
            Loss dictionary
        """
        self.vocoder.train()
        
        # Forward pass
        audio_outputs = self.vocoder(target_mels)
        
        # Compute loss (L1 loss for now, can use more sophisticated losses)
        loss = self.mel_loss(audio_outputs, target_audio)
        
        # Backward pass
        self.vocoder_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vocoder.parameters(), max_norm=1.0)
        self.vocoder_optimizer.step()
        
        return {"vocoder_loss": loss.item()}
    
    def train_epoch(self, train_loader, val_loader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epoch: Epoch number
        
        Returns:
            Metrics dictionary
        """
        train_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            text_input = batch['text'].to(self.device)
            text_lengths = batch['text_lengths'].to(self.device)
            target_mels = batch['mels'].to(self.device)
            target_gates = batch['gates'].to(self.device)
            target_audio = batch['audio'].to(self.device)
            
            # Train steps
            tacotron2_losses = self.train_tacotron2_step(
                text_input, text_lengths, target_mels, target_gates
            )
            vocoder_losses = self.train_vocoder_step(target_mels, target_audio)
            
            losses = {**tacotron2_losses, **vocoder_losses}
            train_losses.append(losses)
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Epoch {epoch} | Batch {batch_idx + 1} | "
                           f"Tacotron2 Loss: {tacotron2_losses['tacotron2_total_loss']:.4f} | "
                           f"Vocoder Loss: {vocoder_losses['vocoder_loss']:.4f}")
        
        # Average losses
        avg_losses = {}
        for key in train_losses[0].keys():
            avg_losses[f"train_{key}"] = np.mean([loss[key] for loss in train_losses])
        
        # Validation
        val_losses = self.validate(val_loader)
        
        return {**avg_losses, **val_losses}
    
    def validate(self, val_loader) -> Dict[str, float]:
        """Validation step"""
        self.tacotron2.eval()
        self.vocoder.eval()
        
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                text_input = batch['text'].to(self.device)
                text_lengths = batch['text_lengths'].to(self.device)
                target_mels = batch['mels'].to(self.device)
                target_gates = batch['gates'].to(self.device)
                target_audio = batch['audio'].to(self.device)
                
                # Tacotron2
                mel_outputs, gate_outputs, _ = self.tacotron2(text_input, text_lengths, target_mels)
                mel_loss = self.mel_loss(mel_outputs, target_mels)
                gate_loss = self.gate_loss(gate_outputs.squeeze(-1), target_gates)
                tacotron2_loss = mel_loss + gate_loss
                
                # Vocoder
                audio_outputs = self.vocoder(target_mels)
                vocoder_loss = self.mel_loss(audio_outputs, target_audio)
                
                val_losses.append({
                    "tacotron2_loss": tacotron2_loss.item(),
                    "vocoder_loss": vocoder_loss.item()
                })
        
        avg_val_losses = {}
        for key in val_losses[0].keys():
            avg_val_losses[f"val_{key}"] = np.mean([loss[key] for loss in val_losses])
        
        return avg_val_losses
    
    def save_checkpoint(self, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'tacotron2_state': self.tacotron2.state_dict(),
            'vocoder_state': self.vocoder.state_dict(),
            'tacotron2_optimizer': self.tacotron2_optimizer.state_dict(),
            'vocoder_optimizer': self.vocoder_optimizer.state_dict(),
            'metrics': metrics
        }
        
        path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.tacotron2.load_state_dict(checkpoint['tacotron2_state'])
            self.vocoder.load_state_dict(checkpoint['vocoder_state'])
            self.tacotron2_optimizer.load_state_dict(checkpoint['tacotron2_optimizer'])
            self.vocoder_optimizer.load_state_dict(checkpoint['vocoder_optimizer'])
            logger.info(f"Checkpoint loaded from {checkpoint_path}")
            return checkpoint['epoch'], checkpoint['metrics']
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return 0, {}
