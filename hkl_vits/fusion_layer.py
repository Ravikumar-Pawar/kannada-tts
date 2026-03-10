# fusion_layer.py

import torch
import torch.nn as nn

class FusionLayer(nn.Module):

    def __init__(self, hidden_dim=256):

        super().__init__()

        self.linear = nn.Linear(hidden_dim*2, hidden_dim)

    def forward(self, Hg, Hp):

        min_len = min(Hg.size(1), Hp.size(1))

        Hg = Hg[:,:min_len,:]
        Hp = Hp[:,:min_len,:]

        H = torch.cat([Hg,Hp],dim=-1)

        H = self.linear(H)

        return H