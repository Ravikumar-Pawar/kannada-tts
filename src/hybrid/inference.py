"""
Advanced Inference Pipeline for Hybrid Approach
With style control, prosody enhancement, and post-processing
"""

import torch
import numpy as np
import logging
import soundfile as sf
from typing import Tuple, Optional, Dict
from .processors import AudioPostProcessor

logger = logging.getLogger(__name__)


class HybridInference:
    """Advanced Inference for Hybrid Tacotron2 + HiFiGAN"""
    
    def __init__(self, tacotron2_model, vocoder_model,
                 character_mapping: Optional[dict] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize hybrid inference
        
        Args:
            tacotron2_model: Enhanced Tacotron2 model
            vocoder_model: Enhanced vocoder model
            character_mapping: Character to index mapping
            device: Device to use
        """
        self.tacotron2 = tacotron2_model.to(device).eval()
        self.vocoder = vocoder_model.to(device).eval()
        self.device = device
        self.sample_rate = 22050
        
        # Post-processor
        self.post_processor = AudioPostProcessor(self.sample_rate)
        
        # Default character mapping for Kannada
        if character_mapping is None:
            self.character_mapping = self._get_default_kannada_mapping()
        else:
            self.character_mapping = character_mapping
        
        logger.info(f"HybridInference initialized on {device}")
    
    def _get_default_kannada_mapping(self) -> dict:
        """Get default Kannada character mapping"""
        kannada_chars = [
            'ಅ', 'ಆ', 'ಇ', 'ಈ', 'ಉ', 'ಊ', 'ಋ', 'ಌ', 'ಎ', 'ಏ', 
            'ಐ', 'ಒ', 'ಓ', 'ಔ', 'ಘ', 'ಙ', 'ಚ', 'ಛ', 'ಜ', 'ಝ',
            'ಞ', 'ಟ', 'ಠ', 'ಡ', 'ಢ', 'ಣ', 'ತ', 'ಥ', 'ದ', 'ಧ',
            'ನ', 'ಪ', 'ಫ', 'ಬ', 'ಭ', 'ಮ', 'ಯ', 'ರ', 'ಲ', 'ವ',
            'ಶ', 'ಷ', 'ಸ', 'ಹ', 'ಾ', 'ಿ', 'ೀ', 'ುೂ', 'ೃ', 'ೆ',
            'ೇ', 'ೈ', 'ೊ', 'ೋ', 'ೌ', 'ೃ', 'ಂ', 'ಃ', '|', ' ',
            '-', '?', '.', ',', '!', ':', ';', '(', ')', '[', ']'
        ]
        return {char: idx for idx, char in enumerate(kannada_chars)}
    
    def text_to_sequence(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert text to character indices"""
        sequence = []
        for char in text:
            if char in self.character_mapping:
                sequence.append(self.character_mapping[char])
        
        if not sequence:
            sequence = [0]
        
        sequence_tensor = torch.LongTensor(sequence).unsqueeze(0).to(self.device)
        length_tensor = torch.LongTensor([len(sequence)]).to(self.device)
        
        return sequence_tensor, length_tensor
    
    def synthesize(self, text: str,
                   reference_audio: Optional[np.ndarray] = None,
                   emotion: str = "neutral",
                   post_processing: str = "advanced",
                   max_mel_steps: int = 1000,
                   gate_threshold: float = 0.5) -> np.ndarray:
        """
        Advanced synthesis with style control and post-processing
        
        Args:
            text: Input text
            reference_audio: Reference audio for style extraction (optional)
            emotion: Emotional tone ("neutral", "happy", "sad", "angry", "surprised")
            post_processing: Post-processing pipeline ("standard", "advanced", "quality", "speed")
            max_mel_steps: Maximum mel generation steps
            gate_threshold: Stop generation threshold
        
        Returns:
            Audio waveform
        """
        logger.info(f"Synthesizing: {text} (emotion={emotion}, pp={post_processing})")
        
        with torch.no_grad():
            # Text processing
            sequence, length = self.text_to_sequence(text)
            
            # Reference mel extraction if provided
            reference_mel = None
            if reference_audio is not None:
                # Convert reference audio to mel spectrogram
                try:
                    import librosa
                    mel = librosa.feature.melspectrogram(
                        y=reference_audio, sr=self.sample_rate, n_mels=80
                    )
                    mel = torch.FloatTensor(mel).unsqueeze(0).to(self.device)
                    reference_mel = mel
                    logger.info("Using reference audio for style extraction")
                except:
                    logger.warning("Could not process reference audio")
            
            # Generate mel spectrogram with style
            mel_outputs, gate_outputs, extra = self.tacotron2(
                sequence, length, mels=None, reference_mel=reference_mel
            )
            
            # Stop at gate threshold
            gates = torch.sigmoid(gate_outputs).squeeze(-1)
            gate_positions = (gates > gate_threshold).nonzero(as_tuple=False)
            
            if gate_positions.numel() > 0:
                stop_idx = min(gate_positions[0].item() + 1, mel_outputs.size(1))
                mel_outputs = mel_outputs[:, :stop_idx, :]
            
            # Get style embedding for vocoder
            style_embedding = extra['style_embedding']
            
            # Generate audio with style
            audio = self.vocoder(mel_outputs, style_embedding)
            
            # Convert to numpy
            audio = audio.squeeze(0).cpu().numpy()
        
        # Post-processing
        if emotion != "neutral":
            audio = self.post_processor.process(
                audio, pipeline=post_processing, emotion=emotion
            )
        else:
            audio = self.post_processor.process(
                audio, pipeline=post_processing, emotion=None
            )
        
        logger.info(f"Generated audio shape: {audio.shape}")
        return audio
    
    def synthesize_batch(self, texts: list,
                        emotion: str = "neutral",
                        post_processing: str = "advanced") -> list:
        """
        Synthesize multiple texts
        
        Args:
            texts: List of texts
            emotion: Emotional tone
            post_processing: Post-processing pipeline
        
        Returns:
            List of audio waveforms
        """
        outputs = []
        for idx, text in enumerate(texts):
            try:
                audio = self.synthesize(
                    text,
                    emotion=emotion,
                    post_processing=post_processing
                )
                outputs.append(audio)
                logger.info(f"Synthesized {idx + 1}/{len(texts)}")
            except Exception as e:
                logger.error(f"Error synthesizing '{text}': {e}")
                outputs.append(None)
        
        return outputs
    
    def save_audio(self, audio: np.ndarray, output_path: str):
        """Save audio to file"""
        sf.write(output_path, audio, self.sample_rate)
        logger.info(f"Audio saved to {output_path}")
    
    def get_synthesis_info(self, text: str) -> Dict:
        """Get synthesis information without generating audio"""
        sequence, length = self.text_to_sequence(text)
        
        return {
            "text": text,
            "sequence_length": length.item(),
            "sample_rate": self.sample_rate,
            "device": str(self.device),
            "model_type": "hybrid"
        }
    
    def set_sample_rate(self, sample_rate: int):
        """Set sample rate and update post-processor"""
        self.sample_rate = sample_rate
        self.post_processor = AudioPostProcessor(sample_rate)
        logger.info(f"Sample rate set to {sample_rate}")
    
    def get_available_emotions(self) -> list:
        """Get list of available emotions"""
        return ["neutral", "happy", "sad", "angry", "surprised"]
    
    def get_available_pipelines(self) -> list:
        """Get list of available post-processing pipelines"""
        return ["standard", "advanced", "quality", "speed"]
