"""
Standard HiFiGAN Vocoder Model for Non-Hybrid Approach
"""

import torch
import torch.nn as nn
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ResBlock(nn.Module):
    """Residual Block for HiFiGAN"""
    
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super(ResBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=dilation)
        self.activation = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        residual = x
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        return x + residual


class VocoderModel(nn.Module):
    """Standard HiFiGAN Vocoder Model"""
    
    def __init__(self, in_channels: int = 80, out_channels: int = 1):
        super(VocoderModel, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.leaky_relu = nn.LeakyReLU(0.1)
        
        # Initial projection
        self.mel_dense = nn.Linear(in_channels, 512)
        
        # Upsampling layers
        self.upsample_scales = [8, 8, 2, 2]
        self.upsample_layers = nn.ModuleList()
        
        for scale in self.upsample_scales:
            self.upsample_layers.append(
                nn.Sequential(
                    nn.ConvTranspose1d(512, 512, kernel_size=scale*2, stride=scale, padding=scale//2),
                    nn.LeakyReLU(0.1)
                )
            )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for _ in range(3):
            self.res_blocks.append(ResBlock(512, kernel_size=3, dilation=1))
            self.res_blocks.append(ResBlock(512, kernel_size=3, dilation=3))
            self.res_blocks.append(ResBlock(512, kernel_size=3, dilation=5))
        
        # Output projection
        self.output_conv = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=7, padding=3),
            nn.LeakyReLU(0.1),
            nn.Conv1d(256, out_channels, kernel_size=7, padding=3),
            nn.Tanh()
        )
        
        logger.info(f"VocoderModel initialized - Input: {in_channels}, Output: {out_channels}")
    
    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            mel_spectrogram: Mel spectrogram (batch_size, mel_steps, 80) or (batch_size, 80, mel_steps)
        
        Returns:
            Audio waveform (batch_size, 1, samples)
        """
        # Ensure mel_spectrogram is (batch, mel_steps, 80)
        if mel_spectrogram.dim() == 3 and mel_spectrogram.size(-1) == self.in_channels:
            x = mel_spectrogram
        else:
            x = mel_spectrogram.transpose(1, 2)
        
        # Initial projection
        x = self.mel_dense(x)  # (batch, mel_steps, 512)
        x = x.transpose(1, 2)  # (batch, 512, mel_steps)
        
        # Upsampling
        for upsample_layer in self.upsample_layers:
            x = upsample_layer(x)
        
        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Output
        x = self.output_conv(x)  # (batch, 1, samples)
        
        return x
    
    def save_model(self, model_path: str):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.state_dict(), model_path)
        logger.info(f"Vocoder model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load model checkpoint"""
        if os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path))
            logger.info(f"Vocoder model loaded from {model_path}")
        else:
            logger.warning(f"Vocoder model file not found: {model_path}")
