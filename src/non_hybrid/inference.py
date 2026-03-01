"""
Standard Inference Pipeline for Non-Hybrid Approach
"""

import torch
import numpy as np
import logging
from typing import Tuple, Optional
import soundfile as sf

logger = logging.getLogger(__name__)


class StandardInference:
    """Standard Inference for Tacotron2 + HiFiGAN"""
    
    def __init__(self, tacotron2_model, vocoder_model,
                 character_mapping: Optional[dict] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize inference
        
        Args:
            tacotron2_model: Acoustic model
            vocoder_model: Vocoder model
            character_mapping: Character to index mapping
            device: Device to use
        """
        if tacotron2_model is None:
            raise ValueError("tacotron2_model must be provided for StandardInference")
        self.tacotron2 = tacotron2_model.to(device).eval()

        # vocoder may be optional; fall back to Griffin-Lim if missing
        if vocoder_model is None:
            self.vocoder = None
            logger.warning("No vocoder provided to StandardInference — using Griffin-Lim fallback")
        else:
            self.vocoder = vocoder_model.to(device).eval()
        self.device = device
        self.sample_rate = 22050
        
        # Default character mapping for Kannada
        if character_mapping is None:
            self.character_mapping = self._get_default_kannada_mapping()
        else:
            self.character_mapping = character_mapping
        
        logger.info(f"StandardInference initialized on {device}")
    
    def _get_default_kannada_mapping(self) -> dict:
        """Dynamically build a mapping covering the Kannada unicode block.
        Ensures all common characters and diacritics are included.
        """
        mapping = {}
        for code in range(0x0C80, 0x0CFF + 1):
            ch = chr(code)
            mapping[ch] = len(mapping)
        for ch in [' ', '-', '?', '.', ',', '!', ':', ';', '(', ')', '[', ']', '|']:
            if ch not in mapping:
                mapping[ch] = len(mapping)
        return mapping
    
    def text_to_sequence(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert text to character indices
        
        Args:
            text: Input text
        
        Returns:
            Character indices tensor, length tensor
        """
        sequence = []
        for char in text:
            if char in self.character_mapping:
                sequence.append(self.character_mapping[char])
            else:
                logger.warning(f"Character not in mapping: {char}")
        
        if not sequence:
            sequence = [0]
        
        sequence_tensor = torch.LongTensor(sequence).unsqueeze(0).to(self.device)
        length_tensor = torch.LongTensor([len(sequence)]).to(self.device)
        
        return sequence_tensor, length_tensor
    
    def synthesize(self, text: str, 
                  max_mel_steps: int = 1000,
                  gate_threshold: float = 0.5) -> np.ndarray:
        """
        Synthesize speech from text
        
        Args:
            text: Input text
            max_mel_steps: Maximum steps for mel generation
            gate_threshold: Stop generation threshold
        
        Returns:
            Audio waveform
        """
        logger.info(f"Synthesizing: {text}")
        
        with torch.no_grad():
            # Convert text to sequence
            sequence, length = self.text_to_sequence(text)
            
            # Generate mel spectrogram
            mel_outputs, gate_outputs, _ = self.tacotron2(sequence, length, mels=None)
            
            # Stop at gate threshold
            gates = torch.sigmoid(gate_outputs).squeeze(-1)
            gate_positions = (gates > gate_threshold).nonzero(as_tuple=False)
            
            if gate_positions.numel() > 0:
                stop_idx = gate_positions[0].item() + 1
                mel_outputs = mel_outputs[:, :stop_idx, :]
            
            # guard against empty mel
            if mel_outputs.numel() == 0 or mel_outputs.size(1) == 0:
                logger.warning("Tacotron2 produced empty mel; returning silence")
                return np.zeros(16000, dtype=np.float32)
            
            # Generate audio from mel using vocoder if available
            if self.vocoder is not None:
                audio = self.vocoder(mel_outputs)
                audio = audio.squeeze(0).cpu().numpy()
            else:
                try:
                    import librosa
                    mel = mel_outputs.squeeze(0).cpu().numpy().T
                    audio = librosa.feature.inverse.mel_to_audio(
                        mel, sr=self.sample_rate, n_fft=2048, hop_length=256
                    )
                    audio = audio.astype(np.float32)
                except Exception:
                    logger.exception("Griffin-Lim fallback failed; returning silence")
                    audio = np.zeros(16000, dtype=np.float32)
            
            logger.info(f"Generated audio shape: {audio.shape}")
        
        return audio
    
    def save_audio(self, audio: np.ndarray, output_path: str):
        """
        Save audio to file
        
        Args:
            audio: Audio waveform
            output_path: Output file path
        """
        sf.write(output_path, audio, self.sample_rate)
        logger.info(f"Audio saved to {output_path}")
    
    def synthesize_batch(self, texts: list) -> list:
        """
        Synthesize multiple texts
        
        Args:
            texts: List of texts
        
        Returns:
            List of audio waveforms
        """
        outputs = []
        for text in texts:
            try:
                audio = self.synthesize(text)
                outputs.append(audio)
            except Exception as e:
                logger.error(f"Error synthesizing '{text}': {e}")
                outputs.append(None)
        
        return outputs
    
    def set_sample_rate(self, sample_rate: int):
        """Set sample rate"""
        self.sample_rate = sample_rate
        logger.info(f"Sample rate set to {sample_rate}")
