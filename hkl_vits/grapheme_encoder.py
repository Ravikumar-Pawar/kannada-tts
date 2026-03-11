# grapheme_encoder.py

import torch
import torch.nn as nn
import math

class GraphemeEncoder(nn.Module):
    """
    Grapheme Encoder for Kannada HKL-VITS
    Converts grapheme sequences to continuous embeddings
    Output: Hg ∈ R^(n × d) where n is sequence length and d is hidden_dim
    """

    def __init__(self, vocab_size, hidden_dim=256, num_layers=4, nhead=4, dropout=0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer - converts token IDs to embeddings
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
        
        # Transformer encoder - captures relationships between graphemes
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, src_key_padding_mask=None):
        """
        Args:
            x: (batch_size, seq_len) - grapheme token indices
            src_key_padding_mask: (batch_size, seq_len) - padding mask
        
        Returns:
            output: (batch_size, seq_len, hidden_dim) - grapheme embeddings Hg
        """
        # Embedding
        x = self.embedding(x) * math.sqrt(self.hidden_dim)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Normalization
        x = self.norm(x)
        
        return x  # Hg ∈ R^(batch_size × n × d)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer (based on Vaswani et al. 2017)
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create constant positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)