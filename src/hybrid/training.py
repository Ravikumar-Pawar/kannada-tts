"""
Advanced Training Pipeline for Hybrid Approach
With style control, duration prediction, and advanced losses
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


class HybridTrainer:
    """Advanced Trainer for Hybrid Tacotron2 + HiFiGAN"""
    
    def __init__(self, tacotron2_model: nn.Module,
                 vocoder_model: nn.Module,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize hybrid trainer
        
        Args:
            tacotron2_model: Enhanced Tacotron2 model
            vocoder_model: Enhanced vocoder model
            device: Device to use
        """
        self.tacotron2 = tacotron2_model.to(device)
        self.vocoder = vocoder_model.to(device)
        self.device = device
        
        # Optimizers with learning rate scheduling
        self.tacotron2_optimizer = optim.Adam(self.tacotron2.parameters(), lr=1e-3, betas=(0.9, 0.999))
        self.vocoder_optimizer = optim.Adam(self.vocoder.parameters(), lr=1e-4, betas=(0.9, 0.999))
        
        # Learning rate schedulers
        self.tacotron2_scheduler = optim.lr_scheduler.ExponentialLR(self.tacotron2_optimizer, gamma=0.99)
        self.vocoder_scheduler = optim.lr_scheduler.ExponentialLR(self.vocoder_optimizer, gamma=0.995)
        
        # Loss functions
        self.mel_loss = nn.L1Loss()
        self.gate_loss = nn.BCEWithLogitsLoss()
        self.duration_loss = nn.L1Loss()
        
        # Checkpoints directory
        self.checkpoint_dir = "output/hybrid_checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        logger.info(f"HybridTrainer initialized on {device}")
    
    def train_tacotron2_step(self, text_input: torch.Tensor,
                            text_lengths: torch.Tensor,
                            target_mels: torch.Tensor,
                            target_gates: torch.Tensor,
                            reference_mels: Optional[torch.Tensor] = None,
                            target_durations: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Single training step for hybrid Tacotron2
        
        Args:
            text_input: Character indices
            text_lengths: Text lengths
            target_mels: Target mel spectrograms
            target_gates: Target gate values
            reference_mels: Reference mels for style extraction
            target_durations: Target phoneme durations
        
        Returns:
            Loss dictionary
        """
        self.tacotron2.train()
        
        # Forward pass
        mel_outputs, gate_outputs, extra_outputs = self.tacotron2(
            text_input, text_lengths, target_mels, reference_mels
        )
        
        # Compute losses
        mel_loss = self.mel_loss(mel_outputs, target_mels)
        gate_loss = self.gate_loss(gate_outputs.squeeze(-1), target_gates)
        
        # Duration prediction loss if available
        duration_loss = torch.tensor(0.0, device=self.device)
        if target_durations is not None and 'durations' in extra_outputs:
            duration_loss = self.duration_loss(
                extra_outputs['durations'].squeeze(-1)[:, :text_input.size(1)],
                target_durations
            )
        
        # Total loss
        total_loss = mel_loss + gate_loss + 0.1 * duration_loss
        
        # Backward pass
        self.tacotron2_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.tacotron2.parameters(), max_norm=1.0)
        self.tacotron2_optimizer.step()
        
        return {
            "tacotron2_mel_loss": mel_loss.item(),
            "tacotron2_gate_loss": gate_loss.item(),
            "tacotron2_duration_loss": duration_loss.item(),
            "tacotron2_total_loss": total_loss.item()
        }
    
    def train_vocoder_step(self, target_mels: torch.Tensor,
                          target_audio: torch.Tensor,
                          style_embedding: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Single training step for hybrid vocoder
        
        Args:
            target_mels: Target mel spectrograms
            target_audio: Target audio waveforms
            style_embedding: Style embeddings for conditional generation
        
        Returns:
            Loss dictionary
        """
        self.vocoder.train()
        
        # Forward pass
        audio_outputs = self.vocoder(target_mels, style_embedding)
        
        # Compute loss
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
            reference_mels = batch.get('reference_mels')
            target_durations = batch.get('durations')
            
            if reference_mels is not None:
                reference_mels = reference_mels.to(self.device)
            if target_durations is not None:
                target_durations = target_durations.to(self.device)
            
            # Train Tacotron2
            tacotron2_losses = self.train_tacotron2_step(
                text_input, text_lengths, target_mels, target_gates,
                reference_mels, target_durations
            )
            
            # Get style embedding for vocoder
            with torch.no_grad():
                _, _, extra = self.tacotron2(text_input, text_lengths, target_mels, reference_mels)
                style = extra['style_embedding']
            
            # Train vocoder
            vocoder_losses = self.train_vocoder_step(target_mels, target_audio, style)
            
            losses = {**tacotron2_losses, **vocoder_losses}
            train_losses.append(losses)
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Epoch {epoch} | Batch {batch_idx + 1} | "
                           f"Tacotron2: {tacotron2_losses['tacotron2_total_loss']:.4f} | "
                           f"Vocoder: {vocoder_losses['vocoder_loss']:.4f}")
        
        # Learning rate scheduling
        self.tacotron2_scheduler.step()
        self.vocoder_scheduler.step()
        
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
                reference_mels = batch.get('reference_mels')
                
                if reference_mels is not None:
                    reference_mels = reference_mels.to(self.device)
                
                # Tacotron2
                mel_outputs, gate_outputs, extra = self.tacotron2(
                    text_input, text_lengths, target_mels, reference_mels
                )
                mel_loss = self.mel_loss(mel_outputs, target_mels)
                gate_loss = self.gate_loss(gate_outputs.squeeze(-1), target_gates)
                tacotron2_loss = mel_loss + gate_loss
                
                # Vocoder
                style = extra['style_embedding']
                audio_outputs = self.vocoder(target_mels, style)
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
            'tacotron2_scheduler': self.tacotron2_scheduler.state_dict(),
            'vocoder_scheduler': self.vocoder_scheduler.state_dict(),
            'metrics': metrics
        }
        
        path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, path)
        logger.info(f"Hybrid checkpoint saved: {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.tacotron2.load_state_dict(checkpoint['tacotron2_state'])
            self.vocoder.load_state_dict(checkpoint['vocoder_state'])
            self.tacotron2_optimizer.load_state_dict(checkpoint['tacotron2_optimizer'])
            self.vocoder_optimizer.load_state_dict(checkpoint['vocoder_optimizer'])
            if 'tacotron2_scheduler' in checkpoint:
                self.tacotron2_scheduler.load_state_dict(checkpoint['tacotron2_scheduler'])
            if 'vocoder_scheduler' in checkpoint:
                self.vocoder_scheduler.load_state_dict(checkpoint['vocoder_scheduler'])
            logger.info(f"Hybrid checkpoint loaded from {checkpoint_path}")
            return checkpoint['epoch'], checkpoint['metrics']
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return 0, {}
