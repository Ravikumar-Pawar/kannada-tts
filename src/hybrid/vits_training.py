"""
VITS Training Pipeline
Variational Inference Text-to-Speech with advanced losses
"""

import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
from datetime import datetime
from typing import Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


class VITSTrainer:
    """VITS Trainer with variational inference"""
    
    def __init__(self, vits_model: nn.Module,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize VITS trainer
        
        Args:
            vits_model: VITS model
            device: Device to use
        """
        self.vits = vits_model.to(device)
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(self.vits.parameters(), lr=1e-4, betas=(0.9, 0.999))
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        # Checkpoints directory
        self.checkpoint_dir = "output/vits_checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        logger.info(f"VITSTrainer initialized on {device}")
    
    def compute_kl_loss(self, mean: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence loss for posterior
        
        Args:
            mean: Mean of posterior (batch, mel_length, hidden_size)
            logstd: Log of std of posterior (batch, mel_length, hidden_size)
        
        Returns:
            KL loss
        """
        kl = 0.5 * torch.mean(mean ** 2 + torch.exp(2 * logstd) - 1 - 2 * logstd)
        return kl
    
    def compute_mel_loss(self, mel_pred: torch.Tensor, mel_target: torch.Tensor) -> torch.Tensor:
        """
        Compute mel spectrogram loss
        
        Args:
            mel_pred: Predicted mel (batch, mel_length, mel_channels)
            mel_target: Target mel (batch, mel_length, mel_channels)
        
        Returns:
            Mel loss
        """
        return self.l1_loss(mel_pred, mel_target)
    
    def compute_duration_loss(self, dur_pred: torch.Tensor, dur_target: torch.Tensor) -> torch.Tensor:
        """
        Compute duration prediction loss
        
        Args:
            dur_pred: Predicted durations (batch, text_length)
            dur_target: Target durations (batch, text_length)
        
        Returns:
            Duration loss
        """
        return self.mse_loss(dur_pred, dur_target)
    
    def train_step(self, text_input: torch.Tensor,
                   text_lengths: torch.Tensor,
                   target_mels: torch.Tensor,
                   target_durations: torch.Tensor) -> Dict[str, float]:
        """
        Single training step
        
        Args:
            text_input: Character indices
            text_lengths: Text lengths
            target_mels: Target mel spectrograms
            target_durations: Target phoneme durations
        
        Returns:
            Loss dictionary
        """
        self.vits.train()
        
        # Forward pass
        outputs = self.vits(text_input, text_lengths, target_mels)
        
        mel_pred = outputs['mel_output']
        mean = outputs['mean']
        logstd = outputs['logstd']
        dur_pred = outputs['durations']
        
        # Compute losses
        mel_loss = self.compute_mel_loss(mel_pred, target_mels)
        kl_loss = self.compute_kl_loss(mean, logstd)
        dur_loss = self.compute_duration_loss(dur_pred, target_durations)
        
        # Total loss
        total_loss = mel_loss + self.vits.kl_weight * kl_loss + 0.1 * dur_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vits.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            "mel_loss": mel_loss.item(),
            "kl_loss": kl_loss.item(),
            "duration_loss": dur_loss.item(),
            "total_loss": total_loss.item()
        }
    
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
            target_durations = batch.get('durations')
            
            if target_durations is not None:
                target_durations = target_durations.to(self.device)
            else:
                # Estimate durations from mel length
                target_durations = torch.ones(text_input.size(0), text_input.size(1), 
                                            device=self.device) * (target_mels.size(1) / text_input.size(1))
            
            # Training step
            losses = self.train_step(text_input, text_lengths, target_mels, target_durations)
            train_losses.append(losses)
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Epoch {epoch} | Batch {batch_idx + 1} | "
                           f"Mel: {losses['mel_loss']:.4f} | "
                           f"KL: {losses['kl_loss']:.4f} | "
                           f"Total: {losses['total_loss']:.4f}")
        
        # Learning rate scheduling
        self.scheduler.step()
        
        # Average training losses
        avg_losses = {}
        for key in train_losses[0].keys():
            avg_losses[f"train_{key}"] = np.mean([loss[key] for loss in train_losses])
        
        # Validation
        val_losses = self.validate(val_loader)
        
        return {**avg_losses, **val_losses}
    
    def validate(self, val_loader) -> Dict[str, float]:
        """Validation step"""
        self.vits.eval()
        
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                text_input = batch['text'].to(self.device)
                text_lengths = batch['text_lengths'].to(self.device)
                target_mels = batch['mels'].to(self.device)
                target_durations = batch.get('durations')
                
                if target_durations is not None:
                    target_durations = target_durations.to(self.device)
                else:
                    target_durations = torch.ones(text_input.size(0), text_input.size(1),
                                                device=self.device) * (target_mels.size(1) / text_input.size(1))
                
                # Forward pass
                outputs = self.vits(text_input, text_lengths, target_mels)
                
                mel_pred = outputs['mel_output']
                mean = outputs['mean']
                logstd = outputs['logstd']
                dur_pred = outputs['durations']
                
                # Compute losses
                mel_loss = self.compute_mel_loss(mel_pred, target_mels)
                kl_loss = self.compute_kl_loss(mean, logstd)
                dur_loss = self.compute_duration_loss(dur_pred, target_durations)
                
                total_loss = mel_loss + self.vits.kl_weight * kl_loss + 0.1 * dur_loss
                
                val_losses.append({
                    "mel_loss": mel_loss.item(),
                    "kl_loss": kl_loss.item(),
                    "duration_loss": dur_loss.item(),
                    "total_loss": total_loss.item()
                })
        
        avg_val_losses = {}
        for key in val_losses[0].keys():
            avg_val_losses[f"val_{key}"] = np.mean([loss[key] for loss in val_losses])
        
        return avg_val_losses
    
    def save_checkpoint(self, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state': self.vits.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'metrics': metrics
        }
        
        path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, path)
        logger.info(f"VITS checkpoint saved: {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.vits.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            logger.info(f"VITS checkpoint loaded from {checkpoint_path}")
            return checkpoint['epoch'], checkpoint['metrics']
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return 0, {}
