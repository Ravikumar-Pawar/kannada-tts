# phoneme_encoder.py

import torch
import torch.nn as nn

class PhonemeEncoder(nn.Module):

    def __init__(self, phoneme_vocab, hidden_dim=256):

        super().__init__()

        self.embedding = nn.Embedding(phoneme_vocab, hidden_dim)

        self.encoder = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, p):

        p = self.embedding(p)

        p,_ = self.encoder(p)

        return p


#Hp ∈ R(m × d)