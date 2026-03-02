"""
VITS-based Inference Pipeline for Hybrid Approach
Advanced synthesis with variational inference and end-to-end quality
"""

import torch
import numpy as np
import logging
import soundfile as sf
import unicodedata
from typing import Tuple, Optional, Dict

logger = logging.getLogger(__name__)


class VITSInference:
    """VITS Inference Engine - End-to-end TTS with superior quality"""
    
    def __init__(self, vits_model,
                 character_mapping: Optional[dict] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize VITS inference
        
        Args:
            vits_model: VITS model instance
            character_mapping: Character to index mapping
            device: Device to use
        """
        self.vits = vits_model.to(device).eval()
        self.device = device
        self.sample_rate = 22050
        
        # Default character mapping for Kannada
        if character_mapping is None:
            self.character_mapping = self._get_default_kannada_mapping()
        else:
            self.character_mapping = character_mapping
        
        logger.info(f"VITSInference initialized on {device}")
    
    def _get_default_kannada_mapping(self) -> dict:
        """Dynamically build a mapping covering the Kannada unicode block.
        This ensures even complex syllables and diacritics are mapped so
        no characters are left unhandled during inference.
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
        
        Kannada text is first normalized to Unicode NFC form so that
diacritic marks and composed syllables are represented consistently.  The
normalized string is then mapped character-by-character using the
`character_mapping` dictionary. Characters not found trigger a logged
warning and are skipped; an entirely empty result falls back to a
placeholder index of 0.

        Args:
            text: Input text
        
        Returns:
            Character indices tensor, length tensor
        """
        # normalization step ensures canonical representation of Kannada
        text = unicodedata.normalize('NFC', text)

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
                   emotion: str = "neutral",
                   post_processing: str = "advanced",
                   temperature: float = 0.667) -> np.ndarray:
        """
        Synthesize speech from text using VITS
        
        Args:
            text: Input text
            emotion: Emotional tone (currently for post-processing)
            post_processing: Post-processing pipeline
            temperature: Sampling temperature for variational inference
        
        Returns:
            Audio waveform
        """
        logger.info(f"VITS Synthesizing: {text}")
        
        with torch.no_grad():
            # Convert text to sequence
            sequence, length = self.text_to_sequence(text)
            
            # VITS forward pass
            outputs = self.vits(sequence, length, mels=None)
            
            mel_output = outputs['mel_output']  # (batch, mel_length, mel_channels)
            # guard against empty mel
            if mel_output.numel() == 0 or mel_output.size(1) == 0:
                logger.warning("VITS produced empty mel; returning silence")
                return np.zeros(16000, dtype=np.float32)
            
            # Post-process mel spectrogram
            mel_output = mel_output.squeeze(0).cpu().numpy()  # (mel_length, mel_channels)
            
            # Vocoding: mel to audio (simple vocoding for demo)
            audio = self._vocoder_simple(mel_output)
            
            logger.info(f"Generated audio shape: {audio.shape}")
        
        return audio
    
    def _vocoder_simple(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """
        Simple Griffin-Lim based vocoding
        
        Args:
            mel_spectrogram: (mel_length, mel_channels)
        
        Returns:
            Audio waveform
        """
        try:
            import librosa
            
            # Convert mel to log magnitude spectrogram
            mel_db = librosa.power_to_db(mel_spectrogram.T, ref=np.max)
            
            # Griffin-Lim reconstruction
            spec = librosa.db_to_power(mel_db)
            audio = librosa.griffinlim(spec, n_iter=50)
            
            return audio
        
        except Exception as e:
            logger.warning(f"Vocoding failed: {e}, returning zeros")
            return np.zeros(mel_spectrogram.shape[0] * 256)
    
    def save_audio(self, audio: np.ndarray, output_path: str):
        """
        Save audio to file
        
        Args:
            audio: Audio waveform
            output_path: Output file path
        """
        sf.write(output_path, audio, self.sample_rate)
        logger.info(f"Audio saved to {output_path}")
    
    def synthesize_batch(self, texts: list, emotion: str = "neutral") -> list:
        """
        Synthesize multiple texts
        
        Args:
            texts: List of texts
            emotion: Emotional tone
        
        Returns:
            List of audio waveforms
        """
        outputs = []
        for text in texts:
            try:
                audio = self.synthesize(text, emotion=emotion)
                outputs.append(audio)
            except Exception as e:
                logger.error(f"Error synthesizing '{text}': {e}")
                outputs.append(None)
        
        return outputs
    
    def set_sample_rate(self, sample_rate: int):
        """Set sample rate"""
        self.sample_rate = sample_rate
        logger.info(f"Sample rate set to {sample_rate}")
    
    def get_available_emotions(self) -> list:
        """Get list of available emotions"""
        return ["neutral", "happy", "sad", "angry", "surprised"]
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "model_type": "VITS",
            "device": str(self.device),
            "sample_rate": self.sample_rate,
            "vocab_size": len(self.character_mapping)
        }
