"""
Enhanced Tacotron2 Model for Hybrid Approach
Features: Multi-head attention, style control, prosody enhancement
"""

import torch
import torch.nn as nn
import os
import logging
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, query_dim: int, memory_dim: int, num_heads: int = 4):
        super(MultiHeadAttention, self).__init__()
        
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.memory_dim = memory_dim
        self.head_dim = query_dim // num_heads
        
        self.query_linear = nn.Linear(query_dim, query_dim)
        self.memory_linear = nn.Linear(memory_dim, query_dim)
        self.value_linear = nn.Linear(memory_dim, query_dim)
        self.out_linear = nn.Linear(query_dim, query_dim)
    
    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (batch, query_dim)
            memory: (batch, seq_len, memory_dim)
        
        Returns:
            context: (batch, query_dim)
        """
        Q = self.query_linear(query).unsqueeze(1)  # (batch, 1, query_dim)
        K = self.memory_linear(memory)  # (batch, seq_len, query_dim)
        V = self.value_linear(memory)  # (batch, seq_len, query_dim)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(1, 2)) / (self.head_dim ** 0.5)
        weights = torch.softmax(scores, dim=-1)  # (batch, 1, seq_len)
        
        # Apply attention
        context = torch.matmul(weights, V).squeeze(1)  # (batch, query_dim)
        context = self.out_linear(context)
        
        return context


class StyleEncoder(nn.Module):
    """Style/Emotion encoder for prosody control"""
    
    def __init__(self, style_dim: int = 128):
        super(StyleEncoder, self).__init__()
        
        self.mel_conv1d = nn.Conv1d(80, 256, kernel_size=3, padding=1)
        self.mel_conv1d_2 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        
        self.lstm = nn.LSTM(256, 128, num_layers=2, batch_first=True, bidirectional=True)
        
        self.style_projection = nn.Linear(256, style_dim)
        
        self.style_dim = style_dim
    
    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_spectrogram: (batch, mel_steps, 80) or (batch, 80, mel_steps)
        
        Returns:
            style_embedding: (batch, style_dim)
        """
        if mel_spectrogram.dim() == 3 and mel_spectrogram.size(-1) == 80:
            x = mel_spectrogram.transpose(1, 2)  # (batch, 80, mel_steps)
        else:
            x = mel_spectrogram
        
        # Conv layers
        x = torch.relu(self.mel_conv1d(x))
        x = torch.relu(self.mel_conv1d_2(x))  # (batch, 256, mel_steps)
        
        # LSTM
        x = x.transpose(1, 2)  # (batch, mel_steps, 256)
        _, (h, c) = self.lstm(x)
        
        # Combine forward and backward
        style_embedding = torch.cat([h[-2], h[-1]], dim=1)  # (batch, 256)
        style_embedding = self.style_projection(style_embedding)  # (batch, style_dim)
        
        return style_embedding


class Tacotron2Hybrid(nn.Module):
    """Enhanced Tacotron2 with multi-head attention and style control"""
    
    def __init__(self, num_chars: int = 132, 
                 encoder_hidden_size: int = 256,
                 decoder_hidden_size: int = 1024,
                 style_dim: int = 128,
                 num_attention_heads: int = 4,
                 prenet_sizes: list = None):
        super(Tacotron2Hybrid, self).__init__()
        
        self.num_chars = num_chars
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.style_dim = style_dim
        
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
        
        # Style encoder
        self.style_encoder = StyleEncoder(style_dim)
        
        # Prenet
        self.prenet_layers = nn.ModuleList()
        input_size = 80
        for prenet_size in prenet_sizes:
            self.prenet_layers.append(nn.Linear(input_size, prenet_size))
            input_size = prenet_size
        
        # Decoder with style modulation
        self.decoder_lstm1 = nn.LSTMCell(
            prenet_sizes[-1] + encoder_hidden_size + style_dim, 
            decoder_hidden_size
        )
        self.decoder_lstm2 = nn.LSTMCell(decoder_hidden_size, decoder_hidden_size)
        
        # Multi-head attention
        self.attention = MultiHeadAttention(decoder_hidden_size, encoder_hidden_size, 
                                           num_attention_heads)
        
        # Output projection
        self.mel_projection = nn.Linear(decoder_hidden_size + style_dim, 80)
        self.gate_projection = nn.Linear(decoder_hidden_size, 1)
        
        # Duration predictor
        self.duration_conv1 = nn.Conv1d(encoder_hidden_size, 256, kernel_size=3, padding=1)
        self.duration_conv2 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.duration_linear = nn.Linear(256, 1)
        
        self.dropout = 0.5
        
        logger.info(f"Tacotron2Hybrid initialized with {num_chars} characters, style_dim={style_dim}")
    
    def predict_durations(self, encoder_outputs: torch.Tensor) -> torch.Tensor:
        """Predict phoneme durations"""
        x = encoder_outputs.transpose(1, 2)  # (batch, encoder_hidden_size, seq_len)
        x = torch.relu(self.duration_conv1(x))
        x = torch.relu(self.duration_conv2(x))
        x = x.transpose(1, 2)  # (batch, seq_len, 256)
        
        durations = torch.relu(self.duration_linear(x))  # (batch, seq_len, 1)
        return durations
    
    def forward(self, text_input: torch.Tensor, 
                text_lengths: torch.Tensor,
                mels: Optional[torch.Tensor] = None,
                reference_mel: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass
        
        Args:
            text_input: Character indices (batch_size, max_text_length)
            text_lengths: Text lengths (batch_size,)
            mels: Mel spectrograms for training (batch_size, mel_steps, 80)
            reference_mel: Reference mel for style extraction (batch_size, mel_steps, 80)
        
        Returns:
            mel_outputs, gate_outputs, extra_outputs (durations, style)
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
        
        # Predict durations
        durations = self.predict_durations(encoder_outputs)
        
        # Extract style
        if reference_mel is not None:
            style_embedding = self.style_encoder(reference_mel)
        else:
            if mels is not None:
                style_embedding = self.style_encoder(mels)
            else:
                batch_size = encoder_outputs.size(0)
                style_embedding = torch.zeros(batch_size, self.style_dim, device=encoder_outputs.device)
        
        # Decoder
        batch_size = encoder_outputs.size(0)
        max_steps = mels.size(1) if mels is not None else 100
        
        mel_outputs = []
        gate_outputs = []
        
        decoder_state1 = torch.zeros(batch_size, self.decoder_hidden_size, device=text_input.device)
        decoder_cell1 = torch.zeros(batch_size, self.decoder_hidden_size, device=text_input.device)
        decoder_state2 = torch.zeros(batch_size, self.decoder_hidden_size, device=text_input.device)
        decoder_cell2 = torch.zeros(batch_size, self.decoder_hidden_size, device=text_input.device)
        
        context = torch.zeros(batch_size, self.encoder_hidden_size, device=text_input.device)
        
        for i in range(max_steps):
            if mels is not None and i < mels.size(1):
                decoder_input = mels[:, i, :]
            else:
                decoder_input = torch.zeros(batch_size, 80, device=text_input.device)
            
            # Prenet
            for layer in self.prenet_layers:
                decoder_input = torch.relu(layer(decoder_input))
                decoder_input = torch.nn.functional.dropout(decoder_input, p=self.dropout, training=self.training)
            
            # Multi-head attention
            context = self.attention(decoder_state2, encoder_outputs)
            
            # LSTM with style
            lstm_input = torch.cat([decoder_input, context, style_embedding], dim=1)
            decoder_state1, decoder_cell1 = self.decoder_lstm1(lstm_input, (decoder_state1, decoder_cell1))
            decoder_state2, decoder_cell2 = self.decoder_lstm2(decoder_state1, (decoder_state2, decoder_cell2))
            
            # Output with style modulation
            output_input = torch.cat([decoder_state2, style_embedding], dim=1)
            mel_output = self.mel_projection(output_input)
            gate_output = self.gate_projection(decoder_state2)
            
            mel_outputs.append(mel_output)
            gate_outputs.append(gate_output)
        
        mel_outputs = torch.stack(mel_outputs, dim=1)  # (batch, steps, 80)
        gate_outputs = torch.stack(gate_outputs, dim=1)  # (batch, steps, 1)
        
        extra_outputs = {
            "durations": durations,
            "style_embedding": style_embedding
        }
        
        return mel_outputs, gate_outputs, extra_outputs
    
    def save_model(self, model_path: str):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.state_dict(), model_path)
        logger.info(f"Hybrid model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load model checkpoint"""
        if os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path))
            logger.info(f"Hybrid model loaded from {model_path}")
        else:
            logger.warning(f"Model file not found: {model_path}")
