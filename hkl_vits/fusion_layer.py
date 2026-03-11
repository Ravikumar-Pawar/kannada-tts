# fusion_layer.py

import torch
import torch.nn as nn

class FusionLayer(nn.Module):
    """
    Fusion Layer for HKL-VITS
    Combines grapheme and phoneme representations using concatenation and linear projection
    
    Mathematical representation:
    H = W * [Hg || Hp] + b
    
    Where:
    - || denotes concatenation
    - W is learnable weight matrix
    - b is bias
    - H is fused representation ∈ R^(batch × n × d)
    """

    def __init__(self, hidden_dim=256, dropout=0.1, fusion_type='linear'):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.fusion_type = fusion_type
        
        if fusion_type == 'linear':
            # Simple linear projection: [Hg || Hp] -> H
            self.linear = nn.Linear(hidden_dim * 2, hidden_dim)
            
        elif fusion_type == 'gated':
            # Gated fusion: α * Hg + (1 - α) * Hp, where α is learned
            self.gate = nn.Linear(hidden_dim * 2, 1)
            
        elif fusion_type == 'attention':
            # Attention-based fusion
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                batch_first=True,
                dropout=dropout
            )
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, Hg, Hp):
        """
        Args:
            Hg: (batch_size, n, hidden_dim) - grapheme embeddings
            Hp: (batch_size, m, hidden_dim) - phoneme embeddings
        
        Returns:
            H: (batch_size, n, hidden_dim) - fused linguistic representation
        """
        # Align sequences to the same length
        min_len = min(Hg.size(1), Hp.size(1))
        Hg = Hg[:, :min_len, :]
        Hp = Hp[:, :min_len, :]
        
        if self.fusion_type == 'linear':
            # Concatenate grapheme and phoneme embeddings
            H = torch.cat([Hg, Hp], dim=-1)
            # Project back to hidden_dim
            H = self.linear(H)
            H = self.dropout(H)
            
        elif self.fusion_type == 'gated':
            # Learn gating weights
            combined = torch.cat([Hg, Hp], dim=-1)
            gate = torch.sigmoid(self.gate(combined))  # (batch, seq_len, 1)
            # Weighted combination
            H = gate * Hg + (1 - gate) * Hp
            H = self.dropout(H)
            
        elif self.fusion_type == 'attention':
            # Use phoneme as query and grapheme as key/value
            H, _ = self.attention(Hp, Hg, Hg)
            H = self.dropout(H)
        
        # Layer normalization
        H = self.norm(H)
        
        return H  # H ∈ R^(batch × n × d)