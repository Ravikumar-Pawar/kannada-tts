"""
Standard Tacotron2 Acoustic Model for Non-Hybrid Approach
"""

import torch
import torch.nn as nn
import os
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class Tacotron2Model(nn.Module):
    """Standard Tacotron2 Model"""
    
    def __init__(self, num_chars: int = 132, 
                 encoder_hidden_size: int = 256,
                 decoder_hidden_size: int = 1024,
                 prenet_sizes: list = None):
        super(Tacotron2Model, self).__init__()
        
        self.num_chars = num_chars
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        
        if prenet_sizes is None:
            prenet_sizes = [256, 256]
        
        # Character embedding
        self.embedding = nn.Embedding(num_chars, 512)
        
        # Encoder
        self.encoder_conv1 = nn.Conv1d(512, encoder_hidden_size, 5, padding=2)
        self.encoder_conv2 = nn.Conv1d(encoder_hidden_size, encoder_hidden_size, 5, padding=2)
        self.encoder_conv3 = nn.Conv1d(encoder_hidden_size, encoder_hidden_size, 5, padding=2)
        self.encoder_lstm = nn.LSTM(encoder_hidden_size, encoder_hidden_size // 2, 
                                    num_layers=2, batch_first=True, bidirectional=True)
        
        # Prenet
        self.prenet_layers = nn.ModuleList()
        input_size = 80
        for prenet_size in prenet_sizes:
            self.prenet_layers.append(nn.Linear(input_size, prenet_size))
            input_size = prenet_size
        
        # Decoder
        self.decoder_lstm1 = nn.LSTMCell(prenet_sizes[-1] + encoder_hidden_size, decoder_hidden_size)
        self.decoder_lstm2 = nn.LSTMCell(decoder_hidden_size, decoder_hidden_size)
        
        # Attention
        self.attention = nn.Linear(decoder_hidden_size + encoder_hidden_size, encoder_hidden_size)
        
        # Output projection
        self.mel_projection = nn.Linear(decoder_hidden_size, 80)
        self.gate_projection = nn.Linear(decoder_hidden_size, 1)
        
        self.dropout = 0.5
        
        logger.info(f"Tacotron2Model initialized with {num_chars} characters")
    
    def forward(self, text_input: torch.Tensor, 
                text_lengths: torch.Tensor,
                mels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            text_input: Character indices (batch_size, max_text_length)
            text_lengths: Text lengths (batch_size,)
            mels: Mel spectrograms for training (batch_size, mel_steps, 80)
        
        Returns:
            mel_outputs, gate_outputs, alignments
        """
        # Embedding
        embedded = self.embedding(text_input)
        
        # Encoder
        x = embedded.transpose(1, 2)  # (batch, 512, seq_len)
        x = torch.relu(self.encoder_conv1(x))
        x = torch.relu(self.encoder_conv2(x))
        x = torch.relu(self.encoder_conv3(x))
        
        x = x.transpose(1, 2)  # (batch, seq_len, 256)
        encoder_outputs, _ = self.encoder_lstm(x)
        
        # Decoder (simplified for non-hybrid)
        batch_size = encoder_outputs.size(0)
        max_steps = mels.size(1) if mels is not None else 100
        
        mel_outputs = []
        gate_outputs = []
        
        decoder_state1 = torch.zeros(batch_size, self.decoder_hidden_size, device=text_input.device)
        decoder_cell1 = torch.zeros(batch_size, self.decoder_hidden_size, device=text_input.device)
        decoder_state2 = torch.zeros(batch_size, self.decoder_hidden_size, device=text_input.device)
        decoder_cell2 = torch.zeros(batch_size, self.decoder_hidden_size, device=text_input.device)
        
        context = torch.zeros(batch_size, self.encoder_hidden_size, device=text_input.device)
        attention_weights = torch.zeros(batch_size, text_input.size(1), device=text_input.device)
        
        for i in range(max_steps):
            if mels is not None and i < mels.size(1):
                decoder_input = mels[:, i, :]
            else:
                decoder_input = torch.zeros(batch_size, 80, device=text_input.device)
            
            # Prenet
            for layer in self.prenet_layers:
                decoder_input = torch.relu(layer(decoder_input))
                decoder_input = torch.nn.functional.dropout(decoder_input, p=self.dropout, training=self.training)
            
            # Attention
            attention_input = torch.cat([decoder_state2, context], dim=1)
            attention_weights = torch.softmax(self.attention(attention_input), dim=-1)
            context = torch.matmul(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
            
            # LSTM
            decoder_state1, decoder_cell1 = self.decoder_lstm1(
                torch.cat([decoder_input, context], dim=1),
                (decoder_state1, decoder_cell1)
            )
            decoder_state2, decoder_cell2 = self.decoder_lstm2(
                decoder_state1,
                (decoder_state2, decoder_cell2)
            )
            
            # Output
            mel_output = self.mel_projection(decoder_state2)
            gate_output = self.gate_projection(decoder_state2)
            
            mel_outputs.append(mel_output)
            gate_outputs.append(gate_output)
        
        mel_outputs = torch.stack(mel_outputs, dim=1)  # (batch, steps, 80)
        gate_outputs = torch.stack(gate_outputs, dim=1)  # (batch, steps, 1)
        
        return mel_outputs, gate_outputs, attention_weights
    
    def save_model(self, model_path: str):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load model checkpoint"""
        if os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path))
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(f"Model file not found: {model_path}")
