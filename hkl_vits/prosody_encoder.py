# prosody_encoder.py

import torch
import torch.nn as nn

class ProsodyEncoder(nn.Module):

    def __init__(self, hidden_dim=256):

        super().__init__()

        self.pitch_embed = nn.Linear(1, hidden_dim)
        self.energy_embed = nn.Linear(1, hidden_dim)

    def forward(self, pitch, energy):

        p = self.pitch_embed(pitch)
        e = self.energy_embed(energy)

        return p + e