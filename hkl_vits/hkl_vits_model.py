# hkl_vits_model.py

import torch
import torch.nn as nn
import math
from typing import Tuple, Dict, Optional

from grapheme_encoder import GraphemeEncoder
from phoneme_encoder import PhonemeEncoder
from fusion_layer import FusionLayer
from prosody_encoder import ProsodyEncoder


class PosteriorEncoder(nn.Module):
    """Posterior encoder for VITS - encodes mel-spectrogram to latent space"""
    
    def __init__(self, n_mels: int, hidden_dim: int, num_layers: int = 4):
        super().__init__()
        
        self.pre_conv = nn.Conv1d(n_mels, hidden_dim, 1)
        
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
                    nn.ReLU()
                )
            )
        self.conv_layers = nn.ModuleList(layers)
        
        self.post_conv = nn.Conv1d(hidden_dim, hidden_dim * 2, 1)
    
    def forward(self, mel_spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mel_spec: (batch, n_mels, time)
        
        Returns:
            Tuple of (mu, log_var) each (batch, hidden_dim, time)
        """
        x = self.pre_conv(mel_spec)
        
        for conv_layer in self.conv_layers:
            x = x + conv_layer(x)  # Skip connection
        
        x = self.post_conv(x)
        
        # Split into mu and log_var
        mu, log_var = torch.chunk(x, 2, dim=1)
        log_var = torch.clamp(log_var, -30, 20)
        
        return mu, log_var


class FlowModel(nn.Module):
    """Flow-based model for sampling from latent space"""
    
    def __init__(self, hidden_dim: int, num_flows: int = 4):
        super().__init__()
        
        self.num_flows = num_flows
        self.flows = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_flows)
        ])
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (batch, hidden_dim, time)
        
        Returns:
            Tuple of (transformed_z, log_det_jacobian)
        """
        log_det = 0
        
        for flow in self.flows:
            z = flow(z)
            log_det = log_det + self._log_det_jacobian(z)
        
        return z, log_det
    
    def _log_det_jacobian(self, z: torch.Tensor) -> torch.Tensor:
        # Simplified - in practice use more sophisticated jacobian computation
        return torch.zeros(z.shape[0], device=z.device)


class Generator(nn.Module):
    """HiFi-GAN-style generator for waveform synthesis"""
    
    def __init__(
        self,
        hidden_dim: int = 256,
        n_fft: int = 1024,
        upsample_scales: list = [8, 8, 2, 2],
        resblock_kernel_sizes: list = [3, 7, 11]
    ):
        super().__init__()
        
        self.num_mels = 80
        
        # Initial projection
        self.pre_conv = nn.Conv1d(hidden_dim, 512, 7, padding=3)
        
        # Upsampling layers
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        
        for i, scale in enumerate(upsample_scales):
            self.ups.append(
                nn.ConvTranspose1d(
                    512 // (2 ** i),
                    512 // (2 ** (i + 1)),
                    kernel_size=scale * 2,
                    stride=scale,
                    padding=scale // 2 + scale % 2,
                    output_padding=scale % 2
                )
            )
            
            resblocks = nn.ModuleList()
            for kernel_size in resblock_kernel_sizes:
                resblocks.append(ResBlock(512 // (2 ** (i + 1)), kernel_size))
            self.resblocks.append(resblocks)
        
        # Final layers
        self.post_conv = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 1, 7, padding=3),
            nn.Tanh()
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, hidden_dim, time)
        
        Returns:
            waveform: (batch, 1, time * upsample_factor)
        """
        x = self.pre_conv(z)
        
        for up, resblocks in zip(self.ups, self.resblocks):
            x = up(x)
            x_res = x
            for resblock in resblocks:
                x = x + resblock(x)
            x = x / math.sqrt(len(resblocks) + 1)
        
        x = self.post_conv(x)
        
        return x


class ResBlock(nn.Module):
    """Residual block for generator"""
    
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, padding=kernel_size // 2
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, padding=kernel_size // 2
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_res = x
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        return x + x_res


class HKLVITS(nn.Module):
    """
    Hybrid Linguistic-Enhanced VITS for Kannada Text-to-Speech
    
    Architecture:
    Text Input → Grapheme/Phoneme Encoders → Fusion Layer → Posterior Encoder 
    → Flow Model → Generator → Waveform
    
    With prosody (pitch, energy) conditioning throughout
    """

    def __init__(
        self,
        vocab_size: int,
        phoneme_vocab: int,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_flows: int = 4,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_mels = n_mels
        
        # Linguistic encoders
        self.grapheme_encoder = GraphemeEncoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.phoneme_encoder = PhonemeEncoder(
            phoneme_vocab=phoneme_vocab,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Fusion layer
        self.fusion = FusionLayer(
            hidden_dim=hidden_dim,
            dropout=dropout,
            fusion_type='linear'
        )
        
        # Prosody encoder
        self.prosody_encoder = ProsodyEncoder(
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Posterior encoder (for training)
        self.posterior_encoder = PosteriorEncoder(
            n_mels=n_mels,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        # Flow model
        self.flow = FlowModel(hidden_dim, num_flows)
        
        # Generator (decoder)
        self.generator = Generator(
            hidden_dim=hidden_dim,
            n_fft=n_fft,
            upsample_scales=[8, 8, 2, 2]
        )
        
        # Projection layer to align linguistic and mel-spec features
        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        text: torch.Tensor,
        phonemes: torch.Tensor,
        pitch: torch.Tensor,
        energy: torch.Tensor,
        mel_target: Optional[torch.Tensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
        phoneme_lengths: Optional[torch.Tensor] = None,
        mel_lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for HKL-VITS
        
        Args:
            text: Grapheme IDs (batch, seq_len)
            phonemes: Phoneme IDs (batch, seq_len)
            pitch: Pitch contour (batch, time, 1)
            energy: Energy contour (batch, time, 1)
            mel_target: Target mel-spectrogram for training (batch, n_mels, time)
            text_lengths: Actual text lengths (batch,)
            phoneme_lengths: Actual phoneme lengths (batch,)
            mel_lengths: Actual mel-spec lengths (batch,)
        
        Returns:
            Dictionary containing model outputs
        """
        batch_size = text.shape[0]
        
        # Create padding masks
        text_mask = None
        if text_lengths is not None:
            text_mask = self._create_mask(text_lengths, text.shape[1])
        
        # Encode linguistic features
        h_grapheme = self.grapheme_encoder(text, src_key_padding_mask=text_mask)
        h_phoneme = self.phoneme_encoder(phonemes, input_lengths=phoneme_lengths)
        
        # Fuse grapheme and phoneme representations
        h_linguistic = self.fusion(h_grapheme, h_phoneme)
        
        # Encode prosody
        h_prosody = self.prosody_encoder(pitch, energy)
        
        # Align prosody to linguistic features
        min_len = min(h_linguistic.shape[1], h_prosody.shape[1])
        h_linguistic = h_linguistic[:, :min_len, :]
        h_prosody = h_prosody[:, :min_len, :]
        
        # Combine linguistic and prosody information
        h_combined = h_linguistic + h_prosody
        h_combined = self.projection(h_combined)
        
        # Forward pass through model
        outputs = {}
        
        if self.training and mel_target is not None:
            # Training: use posterior encoder to get latent distribution
            mu_posterior, log_var_posterior = self.posterior_encoder(mel_target)
            
            # Sample from posterior
            z = self._sample_latent(mu_posterior, log_var_posterior)
            
            outputs['mu_posterior'] = mu_posterior
            outputs['log_var_posterior'] = log_var_posterior
        else:
            # Inference: use linguistic features to approximate latent space
            # Project linguistic features to latent space
            z = h_combined.transpose(1, 2)  # (batch, hidden_dim, time)
        
        # Generate waveform
        waveform = self.generator(z)
        
        # Compute mel-spectrogram from waveform (optional, for verification)
        outputs['waveform'] = waveform
        outputs['z'] = z
        outputs['h_linguistic'] = h_linguistic
        outputs['h_prosody'] = h_prosody
        outputs['h_combined'] = h_combined
        
        return outputs

    def _sample_latent(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor
    ) -> torch.Tensor:
        """Sample from Gaussian distribution using reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def _create_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """Create padding mask"""
        batch_size = lengths.shape[0]
        mask = torch.arange(max_len, device=lengths.device).expand(
            batch_size, max_len
        ) >= lengths.unsqueeze(1)
        return mask

    def inference(
        self,
        text: torch.Tensor,
        phonemes: torch.Tensor,
        pitch: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        temperature: float = 0.667,
        length_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Inference mode - generates speech from text
        
        Args:
            text: Grapheme IDs (batch, seq_len)
            phonemes: Phoneme IDs (batch, seq_len)
            pitch: Optional pitch contour
            energy: Optional energy contour
            temperature: Sampling temperature for latent space
            length_scale: Factor to scale duration
        
        Returns:
            Generated waveform
        """
        self.eval()
        
        with torch.no_grad():
            # Generate default prosody if not provided
            if pitch is None or energy is None:
                h_len = text.shape[1]
                if pitch is None:
                    pitch = torch.ones(text.shape[0], h_len, 1, device=text.device)
                if energy is None:
                    energy = torch.ones(text.shape[0], h_len, 1, device=text.device)
            
            # Forward pass
            outputs = self.forward(text, phonemes, pitch, energy)
            
            # Apply temperature to latent space
            outputs['waveform'] = outputs['waveform'] * temperature
        
        return outputs['waveform']
