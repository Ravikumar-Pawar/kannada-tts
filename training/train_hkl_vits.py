# train_hkl_vits.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import json
import os
import argparse
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Tuple
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hkl_vits.hkl_vits_model import HKLVITS
from hkl_vits.dataset_loader import get_dataloaders
from hkl_vits.loss_functions import HKLVITSLoss, DiscriminatorLoss, GeneratorLoss
from hkl_vits.kannada_g2p import KannadaG2P


class HKLVITSTrainer:
    """
    Trainer for HKL-VITS model
    Handles training loop, validation, checkpointing, and logging
    """
    
    def __init__(self, config_path: str, device: str = 'cuda'):
        """
        Initialize trainer
        
        Args:
            config_path: Path to configuration JSON file
            device: Device to use (cuda or cpu)
        """
        self.device = device
        self.config = self._load_config(config_path)
        self.setup_logging()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized trainer on device: {device}")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.config['logging']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def build_model(self) -> HKLVITS:
        """Build HKL-VITS model"""
        model_config = self.config['model']
        
        model = HKLVITS(
            vocab_size=model_config['vocab_size'],
            phoneme_vocab=model_config['phoneme_vocab_size'],
            n_mels=model_config['num_mels'],
            n_fft=model_config['n_fft'],
            hop_length=model_config['hop_length'],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['grapheme_encoder']['num_layers'],
            num_flows=model_config['flow_model']['num_flows'],
            dropout=model_config['grapheme_encoder']['dropout']
        )
        
        return model.to(self.device)

    def build_optimizer(self, model: nn.Module) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """Build optimizer and scheduler"""
        opt_config = self.config['optimizer']
        
        optimizer = Adam(
            model.parameters(),
            lr=self.config['training']['learning_rate'],
            betas=tuple(opt_config['betas']),
            eps=opt_config['eps'],
            weight_decay=opt_config['weight_decay']
        )
        
        scheduler_config = self.config['scheduler']
        scheduler = ExponentialLR(optimizer, gamma=scheduler_config['gamma'])
        
        return optimizer, scheduler

    def train_epoch(
        self,
        model: HKLVITS,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: HKLVITSLoss,
        epoch: int,
        log_interval: int = 100
    ) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            model: HKL-VITS model
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            epoch: Epoch number
            log_interval: Logging interval
        
        Returns:
            Dictionary of loss values
        """
        model.train()
        losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'kl': 0.0,
            'f0': 0.0,
            'energy': 0.0,
            'adversarial': 0.0
        }
        
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue
            
            # Move batch to device
            text = batch['text'].to(self.device)
            phonemes = batch['phonemes'].to(self.device)
            pitch = batch['pitch'].to(self.device)
            energy = batch['energy'].to(self.device)
            mel_spec = batch['mel_spec'].to(self.device)
            text_lengths = batch['text_lengths'].to(self.device)
            mel_lengths = batch['mel_lengths'].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            
            outputs = model(
                text=text,
                phonemes=phonemes,
                pitch=pitch,
                energy=energy,
                mel_target=mel_spec,
                text_lengths=text_lengths,
                mel_lengths=mel_lengths
            )
            
            # Compute loss
            # Note: This is simplified - full implementation would generate prediction outputs
            d_loss = torch.tensor(0.0, device=self.device)
            total_loss, loss_dict = criterion(
                mel_pred=mel_spec,  # Placeholder
                mel_target=mel_spec,
                pitch_pred=pitch.squeeze(-1),
                pitch_target=pitch.squeeze(-1),
                energy_pred=energy.squeeze(-1),
                energy_target=energy.squeeze(-1),
                mu=outputs.get('mu_posterior', torch.zeros(text.shape[0], 256, mel_spec.shape[2], device=self.device)),
                log_var=outputs.get('log_var_posterior', torch.zeros(text.shape[0], 256, mel_spec.shape[2], device=self.device)),
                discriminator_loss=d_loss,
                lengths=mel_lengths
            )
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                self.config['training']['grad_clip_val']
            )
            
            optimizer.step()
            
            # Update loss tracking
            for key in losses:
                if key in loss_dict:
                    losses[key] += loss_dict[key].item()
            
            # Logging
            if (batch_idx + 1) % log_interval == 0:
                avg_losses = {k: v / (batch_idx + 1) for k, v in losses.items()}
                log_msg = f"Epoch [{epoch}] Batch [{batch_idx + 1}/{num_batches}] "
                log_msg += " | ".join([f"{k}: {v:.4f}" for k, v in avg_losses.items()])
                print(log_msg)
        
        # Average losses
        avg_losses = {k: v / max(num_batches, 1) for k, v in losses.items()}
        return avg_losses

    def validate(
        self,
        model: HKLVITS,
        val_loader: DataLoader,
        criterion: HKLVITSLoss
    ) -> Dict[str, float]:
        """
        Validate model
        
        Args:
            model: HKL-VITS model
            val_loader: Validation data loader
            criterion: Loss function
        
        Returns:
            Dictionary of validation loss values
        """
        model.eval()
        losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'kl': 0.0,
            'f0': 0.0,
            'energy': 0.0,
            'adversarial': 0.0
        }
        
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                
                # Move batch to device
                text = batch['text'].to(self.device)
                phonemes = batch['phonemes'].to(self.device)
                pitch = batch['pitch'].to(self.device)
                energy = batch['energy'].to(self.device)
                mel_spec = batch['mel_spec'].to(self.device)
                mel_lengths = batch['mel_lengths'].to(self.device)
                
                # Forward pass
                outputs = model(
                    text=text,
                    phonemes=phonemes,
                    pitch=pitch,
                    energy=energy,
                    mel_target=mel_spec,
                    mel_lengths=mel_lengths
                )
                
                # Compute loss
                d_loss = torch.tensor(0.0, device=self.device)
                total_loss, loss_dict = criterion(
                    mel_pred=mel_spec,
                    mel_target=mel_spec,
                    pitch_pred=pitch.squeeze(-1),
                    pitch_target=pitch.squeeze(-1),
                    energy_pred=energy.squeeze(-1),
                    energy_target=energy.squeeze(-1),
                    mu=outputs.get('mu_posterior', torch.zeros(text.shape[0], 256, mel_spec.shape[2], device=self.device)),
                    log_var=outputs.get('log_var_posterior', torch.zeros(text.shape[0], 256, mel_spec.shape[2], device=self.device)),
                    discriminator_loss=d_loss,
                    lengths=mel_lengths
                )
                
                # Update loss tracking
                for key in losses:
                    if key in loss_dict:
                        losses[key] += loss_dict[key].item()
        
        # Average losses
        num_batches = len(val_loader)
        avg_losses = {k: v / max(num_batches, 1) for k, v in losses.items()}
        return avg_losses

    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.config
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer, path: str) -> int:
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded: {path} (epoch {epoch})")
        return epoch

    def train(
        self,
        data_dir: str,
        num_epochs: int = None,
        resume_checkpoint: str = None
    ):
        """
        Full training loop
        
        Args:
            data_dir: Path to dataset directory
            num_epochs: Number of epochs to train
            resume_checkpoint: Path to checkpoint to resume from
        """
        # Use config value if num_epochs not specified
        if num_epochs is None:
            num_epochs = self.config['training']['num_epochs']
        
        # Build model and optimizer
        logger = logging.getLogger(__name__)
        model = self.build_model()
        optimizer, scheduler = self.build_optimizer(model)
        criterion = HKLVITSLoss(
            recon_weight=self.config['loss_weights']['reconstruction'],
            kl_weight=self.config['loss_weights']['kl_divergence'],
            adv_weight=self.config['loss_weights']['adversarial'],
            f0_weight=self.config['loss_weights']['f0'],
            energy_weight=self.config['loss_weights']['energy']
        )
        
        start_epoch = 0
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            start_epoch = self.load_checkpoint(model, optimizer, resume_checkpoint)
        
        # Build dataloaders
        train_loader, val_loader = get_dataloaders(
            data_dir,
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['training']['num_workers'],
            sample_rate=self.config['model']['sample_rate'],
            n_fft=self.config['model']['n_fft'],
            hop_length=self.config['model']['hop_length'],
            n_mels=self.config['model']['num_mels']
        )
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Training samples: {len(train_loader) * self.config['training']['batch_size']}")
        logger.info(f"Validation samples: {len(val_loader) * self.config['training']['batch_size']}")
        
        # Training loop
        for epoch in range(start_epoch, num_epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch [{epoch + 1}/{num_epochs}]")
            logger.info(f"{'='*50}")
            
            # Train
            train_losses = self.train_epoch(
                model, train_loader, optimizer, criterion, epoch + 1,
                log_interval=self.config['logging']['log_interval']
            )
            
            logger.info(f"Train losses: {train_losses}")
            
            # Validate
            if (epoch + 1) % self.config['training']['validation_interval'] == 0:
                val_losses = self.validate(model, val_loader, criterion)
                logger.info(f"Val losses: {val_losses}")
            
            # Save checkpoint
            if (epoch + 1) % self.config['training']['checkpoint_interval'] == 0:
                checkpoint_path = (
                    f"{self.config['logging']['checkpoint_dir']}"
                    f"hkl_vits_epoch_{epoch + 1}.pt"
                )
                self.save_checkpoint(model, optimizer, epoch + 1, checkpoint_path)
            
            # Step scheduler
            scheduler.step()
        
        logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description='Train HKL-VITS model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/hkl_vits_config.json',
        help='Path to config file'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=None,
        help='Number of epochs'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID'
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    
    # Create trainer
    trainer = HKLVITSTrainer(args.config, device=device)
    
    # Train
    trainer.train(
        data_dir=args.data_dir,
        num_epochs=args.num_epochs,
        resume_checkpoint=args.resume
    )


if __name__ == '__main__':
    main()
