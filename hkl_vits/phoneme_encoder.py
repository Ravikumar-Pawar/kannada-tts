# phoneme_encoder.py

import torch
import torch.nn as nn
import math

class PhonemeEncoder(nn.Module):
    """
    Phoneme Encoder for Kannada HKL-VITS
    Converts phoneme sequences to continuous embeddings using bidirectional LSTM
    Output: Hp ∈ R^(m × d) where m is sequence length and d is hidden_dim
    """

    def __init__(self, phoneme_vocab, hidden_dim=256, num_layers=2, dropout=0.1, bidirectional=True):
        super().__init__()
        
        self.phoneme_vocab = phoneme_vocab
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Embedding layer - converts phoneme IDs to embeddings
        self.embedding = nn.Embedding(phoneme_vocab, hidden_dim)
        
        # Bidirectional LSTM encoder
        self.encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        # If bidirectional, project back to hidden_dim
        if bidirectional:
            self.projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, p, input_lengths=None):
        """
        Args:
            p: (batch_size, seq_len) - phoneme token indices
            input_lengths: (batch_size,) - actual lengths of sequences (for packing)
        
        Returns:
            output: (batch_size, seq_len, hidden_dim) - phoneme embeddings Hp
        """
        # Embedding with dropout
        p = self.embedding(p)
        p = self.dropout(p)
        
        # Pack padded sequences if lengths provided
        if input_lengths is not None:
            p_packed = nn.utils.rnn.pack_padded_sequence(
                p, input_lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.encoder(p_packed)
            p, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            p, _ = self.encoder(p)
        
        # Project bidirectional output back to hidden_dim if needed
        if self.bidirectional:
            p = self.projection(p)
        
        # Normalization
        p = self.norm(p)
        
        return p  # Hp ∈ R^(batch_size × m × d)