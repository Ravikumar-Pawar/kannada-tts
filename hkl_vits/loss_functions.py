# loss_functions.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class HKLVITSLoss(nn.Module):
    """
    Combined loss function for HKL-VITS training
    
    Total Loss = α_recon * L_reconstruction 
              + α_kl * L_KL 
              + α_adv * L_adversarial 
              + α_f0 * L_F0 
              + α_energy * L_energy
    
    Where:
    - L_reconstruction: L1/MSE loss between predicted and target mel-spectrograms
    - L_KL: KL divergence for latent space regularization
    - L_adversarial: GAN loss for naturalness
    - L_F0: Pitch accuracy loss
    - L_energy: Energy/loudness accuracy loss
    """
    
    def __init__(
        self,
        recon_weight: float = 1.0,
        kl_weight: float = 0.1,
        adv_weight: float = 1.0,
        f0_weight: float = 0.5,
        energy_weight: float = 0.1,
        use_l1: bool = True
    ):
        super().__init__()
        
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.adv_weight = adv_weight
        self.f0_weight = f0_weight
        self.energy_weight = energy_weight
        self.use_l1 = use_l1
        
        # Define individual loss functions
        if use_l1:
            self.recon_loss_fn = nn.L1Loss()
        else:
            self.recon_loss_fn = nn.MSELoss()
        
        self.f0_loss_fn = nn.L1Loss()
        self.energy_loss_fn = nn.L1Loss()

    def forward(
        self,
        mel_pred: torch.Tensor,
        mel_target: torch.Tensor,
        pitch_pred: torch.Tensor,
        pitch_target: torch.Tensor,
        energy_pred: torch.Tensor,
        energy_target: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        discriminator_loss: torch.Tensor = None,
        lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss
        
        Args:
            mel_pred: Predicted mel-spectrogram (batch, n_mels, time)
            mel_target: Target mel-spectrogram (batch, n_mels, time)
            pitch_pred: Predicted pitch (batch, time)
            pitch_target: Target pitch (batch, time)
            energy_pred: Predicted energy (batch, time)
            energy_target: Target energy (batch, time)
            mu: Mean of latent distribution (batch, latent_dim)
            log_var: Log variance of latent distribution (batch, latent_dim)
            discriminator_loss: Loss from discriminator
            lengths: Actual lengths of sequences for masking
        
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        losses = {}
        
        # 1. Reconstruction Loss (Mel-Spectrogram)
        # Mask padding positions if lengths provided
        if lengths is not None:
            mask = self._create_time_mask(mel_target, lengths)
            mel_pred_masked = mel_pred * mask
            mel_target_masked = mel_target * mask
            recon_loss = self.recon_loss_fn(mel_pred_masked, mel_target_masked)
        else:
            recon_loss = self.recon_loss_fn(mel_pred, mel_target)
        
        losses['reconstruction'] = recon_loss
        
        # 2. KL Divergence Loss (Latent Space Regularization)
        # KL(N(μ, σ²) || N(0, 1)) = 0.5 * Σ(μ² + σ² - log(σ²) - 1)
        kl_loss = self._kl_divergence_loss(mu, log_var)
        losses['kl'] = kl_loss
        
        # 3. F0 (Pitch) Loss
        if lengths is not None:
            mask = self._create_time_mask(pitch_target.unsqueeze(1), lengths)
            pitch_pred_masked = pitch_pred * mask.squeeze(1)
            pitch_target_masked = pitch_target * mask.squeeze(1)
            f0_loss = self.f0_loss_fn(pitch_pred_masked, pitch_target_masked)
        else:
            f0_loss = self.f0_loss_fn(pitch_pred, pitch_target)
        
        losses['f0'] = f0_loss
        
        # 4. Energy Loss
        if lengths is not None:
            mask = self._create_time_mask(energy_target.unsqueeze(1), lengths)
            energy_pred_masked = energy_pred * mask.squeeze(1)
            energy_target_masked = energy_target * mask.squeeze(1)
            energy_loss = self.energy_loss_fn(energy_pred_masked, energy_target_masked)
        else:
            energy_loss = self.energy_loss_fn(energy_pred, energy_target)
        
        losses['energy'] = energy_loss
        
        # 5. Adversarial Loss (from GAN discriminator)
        if discriminator_loss is not None:
            losses['adversarial'] = discriminator_loss
            adv_loss = discriminator_loss
        else:
            adv_loss = torch.tensor(0.0, device=mel_pred.device)
            losses['adversarial'] = adv_loss
        
        # Compute weighted total loss
        total_loss = (
            self.recon_weight * recon_loss +
            self.kl_weight * kl_loss +
            self.adv_weight * adv_loss +
            self.f0_weight * f0_loss +
            self.energy_weight * energy_loss
        )
        
        losses['total'] = total_loss
        
        return total_loss, losses

    def _kl_divergence_loss(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence loss: KL(N(μ, σ²) || N(0, 1))
        = 0.5 * Σ(μ² + exp(log_var) - log_var - 1)
        """
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        )
        return kl_loss

    def _create_time_mask(self, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Create binary mask for variable-length sequences
        
        Args:
            target: Target tensor (batch, channels, time) or (batch, time)
            lengths: Actual lengths (batch,)
        
        Returns:
            Mask tensor with same shape as target
        """
        batch_size = target.shape[0]
        max_len = target.shape[-1]
        
        mask = torch.arange(max_len, device=target.device).expand(
            batch_size, max_len
        ) < lengths.unsqueeze(1)
        
        # Add channel dimension if needed
        if target.dim() == 3:
            mask = mask.unsqueeze(1)
        
        return mask.float()


class DiscriminatorLoss(nn.Module):
    """
    GAN Discriminator Loss
    Uses least-squares GAN (LSGAN) for training stability
    """
    
    def __init__(self, loss_type: str = 'lsgan'):
        super().__init__()
        self.loss_type = loss_type
    
    def forward(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute discriminator loss
        
        Args:
            real_pred: Discriminator predictions on real data
            fake_pred: Discriminator predictions on fake (generated) data
        
        Returns:
            Discriminator loss
        """
        if self.loss_type == 'lsgan':
            # LSGAN: E[(D(x) - 1)²] + E[(D(G(z)))²]
            real_loss = F.mse_loss(real_pred, torch.ones_like(real_pred))
            fake_loss = F.mse_loss(fake_pred, torch.zeros_like(fake_pred))
        elif self.loss_type == 'standard':
            # Standard GAN: -E[log(D(x))] - E[log(1 - D(G(z)))]
            real_loss = F.binary_cross_entropy_with_logits(
                real_pred, torch.ones_like(real_pred)
            )
            fake_loss = F.binary_cross_entropy_with_logits(
                fake_pred, torch.zeros_like(fake_pred)
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return real_loss + fake_loss


class GeneratorLoss(nn.Module):
    """
    GAN Generator Loss
    Makes generated samples realistic according to discriminator
    """
    
    def __init__(self, loss_type: str = 'lsgan'):
        super().__init__()
        self.loss_type = loss_type
    
    def forward(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute generator loss
        
        Args:
            fake_pred: Discriminator predictions on fake (generated) data
        
        Returns:
            Generator loss
        """
        if self.loss_type == 'lsgan':
            # LSGAN: E[(D(G(z)) - 1)²]
            loss = F.mse_loss(fake_pred, torch.ones_like(fake_pred))
        elif self.loss_type == 'standard':
            # Standard GAN: -E[log(D(G(z)))]
            loss = F.binary_cross_entropy_with_logits(
                fake_pred, torch.ones_like(fake_pred)
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss


class FocusedMaskedLoss(nn.Module):
    """
    Focused loss that emphasizes important timesteps
    (e.g., phoneme boundaries, pitch changes)
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        focus_mask: torch.Tensor,
        base_loss_fn=None
    ) -> torch.Tensor:
        """
        Compute focused loss
        
        Args:
            pred: Predictions (batch, time, features)
            target: Targets (batch, time, features)
            focus_mask: Binary mask for important timesteps (batch, time, 1)
            base_loss_fn: Base loss function (default: L1)
        
        Returns:
            Focused loss
        """
        if base_loss_fn is None:
            base_loss_fn = nn.L1Loss(reduction='none')
        
        loss = base_loss_fn(pred, target)
        
        # Apply focus mask
        weighted_loss = loss * focus_mask
        
        if self.reduction == 'mean':
            return weighted_loss.sum() / (focus_mask.sum() + 1e-8)
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss
