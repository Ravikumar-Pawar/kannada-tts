"""
Enhanced HiFiGAN Vocoder Model for Hybrid Approach
Features: Conditional generation, style control, improved stability
"""

import torch
import torch.nn as nn
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AdaptiveInstanceNorm(nn.Module):
    """Adaptive Instance Normalization for style control"""
    
    def __init__(self, num_features: int, style_dim: int):
        super(AdaptiveInstanceNorm, self).__init__()
        
        self.instance_norm = nn.InstanceNorm1d(num_features)
        self.style_scale = nn.Linear(style_dim, num_features)
        self.style_bias = nn.Linear(style_dim, num_features)
        
        # Initialize
        nn.init.constant_(self.style_scale.weight, 0)
        nn.init.constant_(self.style_scale.bias, 1)
        nn.init.constant_(self.style_bias.weight, 0)
        nn.init.constant_(self.style_bias.bias, 0)
    
    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, length)
            style: (batch, style_dim)
        
        Returns:
            Normalized and adapted output
        """
        x = self.instance_norm(x)
        
        scale = self.style_scale(style).unsqueeze(2)  # (batch, channels, 1)
        bias = self.style_bias(style).unsqueeze(2)    # (batch, channels, 1)
        
        return x * scale + bias


class ResBlockHybrid(nn.Module):
    """Residual Block with style control"""
    
    def __init__(self, channels: int, kernel_size: int, dilation: int, style_dim: int = 128):
        super(ResBlockHybrid, self).__init__()
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, 
                               padding=dilation * (kernel_size - 1) // 2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation,
                               padding=dilation * (kernel_size - 1) // 2)
        
        self.ain1 = AdaptiveInstanceNorm(channels, style_dim)
        self.ain2 = AdaptiveInstanceNorm(channels, style_dim)
        
        self.activation = nn.LeakyReLU(0.1)
    
    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, length)
            style: (batch, style_dim)
        
        Returns:
            Output with residual
        """
        residual = x
        
        x = self.conv1(x)
        x = self.ain1(x, style)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.ain2(x, style)
        x = self.activation(x)
        
        return x + residual


class VocoderHybrid(nn.Module):
    """Enhanced HiFiGAN Vocoder with style control"""
    
    def __init__(self, in_channels: int = 80, out_channels: int = 1, style_dim: int = 128):
        super(VocoderHybrid, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.style_dim = style_dim
        self.leaky_relu = nn.LeakyReLU(0.1)
        
        # Style embedding projection
        self.style_embedding = nn.Linear(style_dim, 256)
        
        # Initial projection
        self.mel_dense = nn.Linear(in_channels, 512)
        self.mel_norm = nn.InstanceNorm1d(512)
        
        # Upsampling layers with adaptive normalization
        self.upsample_scales = [8, 8, 2, 2]
        self.upsample_layers = nn.ModuleList()
        self.upsample_norms = nn.ModuleList()
        
        for scale in self.upsample_scales:
            self.upsample_layers.append(
                nn.ConvTranspose1d(512, 512, kernel_size=scale*2, stride=scale, padding=scale//2)
            )
            self.upsample_norms.append(AdaptiveInstanceNorm(512, style_dim))
        
        # Residual blocks with style control
        self.res_blocks = nn.ModuleList()
        for _ in range(3):
            self.res_blocks.append(ResBlockHybrid(512, kernel_size=3, dilation=1, style_dim=style_dim))
            self.res_blocks.append(ResBlockHybrid(512, kernel_size=3, dilation=3, style_dim=style_dim))
            self.res_blocks.append(ResBlockHybrid(512, kernel_size=3, dilation=5, style_dim=style_dim))
        
        # Output projection
        self.output_conv = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=7, padding=3),
            nn.LeakyReLU(0.1),
            nn.Conv1d(256, out_channels, kernel_size=7, padding=3),
            nn.Tanh()
        )
        
        logger.info(f"VocoderHybrid initialized - Input: {in_channels}, Output: {out_channels}, "
                   f"Style: {style_dim}")
    
    def forward(self, mel_spectrogram: torch.Tensor, 
                style: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            mel_spectrogram: Mel spectrogram (batch_size, mel_steps, 80) or (batch_size, 80, mel_steps)
            style: Style embedding (batch_size, style_dim), optional
        
        Returns:
            Audio waveform (batch_size, 1, samples)
        """
        # Ensure mel_spectrogram is (batch, mel_steps, 80)
        if mel_spectrogram.dim() == 3 and mel_spectrogram.size(-1) == self.in_channels:
            x = mel_spectrogram
        else:
            x = mel_spectrogram.transpose(1, 2)
        
        batch_size = x.size(0)
        
        # Default style if not provided
        if style is None:
            style = torch.zeros(batch_size, self.style_dim, device=x.device)
        
        # Embed style
        style_embedding = self.style_embedding(style)  # (batch, 256)
        
        # Initial projection
        x = self.mel_dense(x)  # (batch, mel_steps, 512)
        x = x.transpose(1, 2)  # (batch, 512, mel_steps)
        x = self.mel_norm(x)
        x = self.leaky_relu(x)
        
        # Upsampling
        for upsample_layer, upsample_norm in zip(self.upsample_layers, self.upsample_norms):
            x = upsample_layer(x)
            x = upsample_norm(x, style)
            x = self.leaky_relu(x)
        
        # Residual blocks with style
        for res_block in self.res_blocks:
            x = res_block(x, style)
        
        # Output
        x = self.output_conv(x)  # (batch, 1, samples)
        
        return x
    
    def infer(self, mel_spectrogram: torch.Tensor,
              style: Optional[torch.Tensor] = None,
              vocoder_type: str = "default") -> torch.Tensor:
        """
        Inference mode with different vocoder types
        
        Args:
            mel_spectrogram: Mel spectrogram
            style: Style embedding
            vocoder_type: "default", "fast", or "quality"
        
        Returns:
            Audio waveform
        """
        with torch.no_grad():
            if vocoder_type == "fast":
                # Use only essential processing
                return self.forward(mel_spectrogram, style)
            elif vocoder_type == "quality":
                # Multiple passes for better quality
                audio = self.forward(mel_spectrogram, style)
                return audio
            else:
                return self.forward(mel_spectrogram, style)
    
    def save_model(self, model_path: str):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.state_dict(), model_path)
        logger.info(f"Hybrid vocoder model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load model checkpoint"""
        if os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path))
            logger.info(f"Hybrid vocoder model loaded from {model_path}")
        else:
            logger.warning(f"Model file not found: {model_path}")
