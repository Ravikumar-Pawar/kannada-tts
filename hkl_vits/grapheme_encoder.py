# grapheme_encoder.py

import torch
import torch.nn as nn

class GraphemeEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_dim=256):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4
            ),
            num_layers=4
        )

    def forward(self, x):

        x = self.embedding(x)

        x = self.encoder(x)

        return x

#ouput 
# Hg ∈ R(n × d)