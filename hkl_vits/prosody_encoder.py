# prosody_encoder.py

import torch
import torch.nn as nn

class ProsodyEncoder(nn.Module):
    """
    Prosody Encoder for HKL-VITS
    Encodes pitch (F0) and energy features to enhance prosody in speech synthesis
    
    Prosody features:
    - Pitch (F0): Fundamental frequency variations
    - Energy: Overall loudness/intensity of speech
    
    Output combines pitch and energy embeddings for prosodic conditioning
    """

    def __init__(self, hidden_dim=256, num_pitch_bins=256, num_energy_bins=256, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Pitch embedding and processing
        self.pitch_embedding = nn.Embedding(num_pitch_bins, hidden_dim)
        self.pitch_embed_layer = nn.Linear(1, hidden_dim)
        self.pitch_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        # Energy embedding and processing
        self.energy_embedding = nn.Embedding(num_energy_bins, hidden_dim)
        self.energy_embed_layer = nn.Linear(1, hidden_dim)
        self.energy_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        # Fusion of pitch and energy
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Dropout and normalization
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, pitch, energy, pitch_bins=None, energy_bins=None):
        """
        Args:
            pitch: (batch_size, seq_len, 1) - continuous pitch values (F0)
            energy: (batch_size, seq_len, 1) - continuous energy values
            pitch_bins: (batch_size, seq_len) - discretized pitch bins (optional)
            energy_bins: (batch_size, seq_len) - discretized energy bins (optional)
        
        Returns:
            prosody: (batch_size, seq_len, hidden_dim) - prosody embedding
        """
        # Process pitch
        if pitch_bins is not None:
            # Use discrete pitch embedding
            p_embed = self.pitch_embedding(pitch_bins)  # (batch, seq_len, hidden_dim)
        else:
            # Use continuous pitch values
            p_embed = self.pitch_embed_layer(pitch)  # (batch, seq_len, hidden_dim)
        
        # Apply convolution and activation
        p_embed = torch.relu(self.pitch_conv(p_embed.transpose(1, 2)).transpose(1, 2))
        p_embed = self.dropout(p_embed)
        
        # Process energy
        if energy_bins is not None:
            # Use discrete energy embedding
            e_embed = self.energy_embedding(energy_bins)  # (batch, seq_len, hidden_dim)
        else:
            # Use continuous energy values
            e_embed = self.energy_embed_layer(energy)  # (batch, seq_len, hidden_dim)
        
        # Apply convolution and activation
        e_embed = torch.relu(self.energy_conv(e_embed.transpose(1, 2)).transpose(1, 2))
        e_embed = self.dropout(e_embed)
        
        # Fuse pitch and energy representations
        prosody = torch.cat([p_embed, e_embed], dim=-1)  # (batch, seq_len, hidden_dim * 2)
        prosody = self.fusion(prosody)  # (batch, seq_len, hidden_dim)
        
        # Normalize
        prosody = self.norm(prosody)
        
        return prosody  # prosody ∈ R^(batch × seq_len × hidden_dim)