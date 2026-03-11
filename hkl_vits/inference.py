# inference.py

import torch
import torchaudio
import numpy as np
import json
from pathlib import Path
from typing import Optional, Tuple
import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hkl_vits.hkl_vits_model import HKLVITS
from hkl_vits.kannada_g2p import KannadaG2P


class HKLVITSInference:
    """
    Inference module for HKL-VITS
    Generates speech from Kannada text
    """
    
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = 'cuda'
    ):
        """
        Initialize inference engine
        
        Args:
            config_path: Path to config JSON file
            checkpoint_path: Path to model checkpoint
            device: Device to use (cuda or cpu)
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.config = self._load_config(config_path)
        self.g2p = KannadaG2P()
        
        # Build model
        self.model = self._build_model()
        
        # Load checkpoint
        self._load_checkpoint(checkpoint_path)
        
        self.model.eval()
        print(f"Model loaded on device: {self.device}")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config

    def _build_model(self) -> HKLVITS:
        """Build model"""
        model_config = self.config['model']
        
        model = HKLVITS(
            vocab_size=model_config['vocab_size'],
            phoneme_vocab=model_config['phoneme_vocab_size'],
            n_mels=model_config['num_mels'],
            n_fft=model_config['n_fft'],
            hop_length=model_config['hop_length'],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['grapheme_encoder']['num_layers'],
            num_flows=model_config['flow_model']['num_flows'],
            dropout=model_config['grapheme_encoder']['dropout']
        )
        
        return model.to(self.device)

    def _load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint loaded: {checkpoint_path}")

    def preprocess_text(self, kannada_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess Kannada text to model inputs
        
        Args:
            kannada_text: Kannada text string
        
        Returns:
            Tuple of (text_ids, phoneme_ids) tensors
        """
        # Convert text to grapheme IDs
        grapheme_ids = torch.tensor(
            [ord(c) for c in kannada_text if ord(c) >= 2944 and ord(c) <= 3007],  # Kannada range
            dtype=torch.long
        ).unsqueeze(0)  # Add batch dimension
        
        # Convert text to phoneme IDs
        phoneme_ids, _ = self.g2p.batch_text_to_phoneme_ids([kannada_text])
        
        return grapheme_ids.to(self.device), phoneme_ids.to(self.device)

    def generate_prosody(
        self,
        length: int,
        pitch_mean: float = 150.0,
        pitch_std: float = 30.0,
        energy_mean: float = 0.0,
        energy_std: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate prosody features (pitch and energy)
        
        Args:
            length: Sequence length
            pitch_mean: Mean pitch in Hz
            pitch_std: Std dev of pitch
            energy_mean: Mean energy
            energy_std: Std dev of energy
        
        Returns:
            Tuple of (pitch, energy) tensors
        """
        # Generate pitch contour (with some variations)
        pitch = torch.randn(1, length) * pitch_std + pitch_mean
        pitch = torch.clamp(pitch, min=self.config['model']['f0_min'], max=self.config['model']['f0_max'])
        pitch = pitch.unsqueeze(-1).to(self.device)  # (1, length, 1)
        
        # Generate energy contour
        energy = torch.randn(1, length) * energy_std + energy_mean
        energy = energy.unsqueeze(-1).to(self.device)  # (1, length, 1)
        
        # Smooth prosody with moving average for naturalness
        window_size = 5
        pitch_smoothed = torch.nn.functional.avg_pool1d(
            pitch.transpose(1, 2),
            kernel_size=window_size,
            stride=1,
            padding=window_size // 2
        ).transpose(1, 2)
        
        energy_smoothed = torch.nn.functional.avg_pool1d(
            energy.transpose(1, 2),
            kernel_size=window_size,
            stride=1,
            padding=window_size // 2
        ).transpose(1, 2)
        
        return pitch_smoothed, energy_smoothed

    def synthesize(
        self,
        kannada_text: str,
        temperature: float = 0.667,
        length_scale: float = 1.0,
        pitch: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Synthesize speech from Kannada text
        
        Args:
            kannada_text: Kannada text to synthesize
            temperature: Sampling temperature
            length_scale: Duration scale factor
            pitch: Optional custom pitch contour
            energy: Optional custom energy contour
            save_path: Optional path to save audio file
        
        Returns:
            Generated waveform as numpy array
        """
        with torch.no_grad():
            # Preprocess text
            text_ids, phoneme_ids = self.preprocess_text(kannada_text)
            
            # Generate prosody if not provided
            if pitch is None or energy is None:
                pitch, energy = self.generate_prosody(text_ids.shape[1])
            
            # Forward pass (inference mode)
            waveform = self.model.inference(
                text=text_ids,
                phonemes=phoneme_ids,
                pitch=pitch,
                energy=energy,
                temperature=temperature,
                length_scale=length_scale
            )
            
            # Convert to numpy and squeeze
            waveform = waveform.squeeze(0).cpu().numpy()
            
            # Normalize audio
            max_val = np.abs(waveform).max()
            if max_val > 1.0:
                waveform = waveform / max_val
            
            # Save audio if path provided
            if save_path:
                self._save_audio(waveform, save_path)
            
            return waveform

    def _save_audio(self, waveform: np.ndarray, save_path: str):
        """Save audio to file"""
        sample_rate = self.config['model']['sample_rate']
        waveform_tensor = torch.from_numpy(waveform).float()
        torchaudio.save(save_path, waveform_tensor.unsqueeze(0), sample_rate)
        print(f"Audio saved: {save_path}")

    def synthesize_batch(
        self,
        texts: list,
        save_dir: Optional[str] = None
    ) -> list:
        """
        Synthesize batch of texts
        
        Args:
            texts: List of Kannada text strings
            save_dir: Optional directory to save audio files
        
        Returns:
            List of waveforms
        """
        waveforms = []
        
        for i, text in enumerate(texts):
            print(f"Synthesizing text {i+1}/{len(texts)}: {text}")
            
            save_path = None
            if save_dir:
                save_path = os.path.join(save_dir, f"output_{i+1:03d}.wav")
            
            waveform = self.synthesize(text, save_path=save_path)
            waveforms.append(waveform)
        
        return waveforms


class InteractiveInference:
    """Interactive interface for HKL-VITS inference"""
    
    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'cuda'):
        """Initialize interactive interface"""
        self.inference_engine = HKLVITSInference(config_path, checkpoint_path, device)

    def run(self, output_dir: str = 'outputs'):
        """Run interactive synthesis loop"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("HKL-VITS Kannada Text-to-Speech Interactive Mode")
        print("="*60)
        print("Commands:")
        print("  'q' / 'quit': Exit")
        print("  'p PITCH': Set pitch (Hz)")
        print("  'e ENERGY': Set energy")
        print("  'l LENGTH': Set length scale")
        print("  't TEMP': Set temperature")
        print("  Any other text: Synthesize speech")
        print("="*60 + "\n")
        
        pitch = None
        energy = None
        length_scale = 1.0
        temperature = 0.667
        counter = 0
        
        while True:
            try:
                user_input = input("Enter Kannada text or command: ").strip()
                
                if not user_input:
                    continue
                
                # Process commands
                if user_input.lower() in ['q', 'quit']:
                    print("Exiting...")
                    break
                
                elif user_input.lower().startswith('p '):
                    try:
                        pitch_val = float(user_input[2:])
                        print(f"Pitch set to {pitch_val} Hz")
                    except ValueError:
                        print("Invalid pitch value")
                    continue
                
                elif user_input.lower().startswith('e '):
                    try:
                        energy_val = float(user_input[2:])
                        print(f"Energy set to {energy_val}")
                    except ValueError:
                        print("Invalid energy value")
                    continue
                
                elif user_input.lower().startswith('l '):
                    try:
                        length_scale = float(user_input[2:])
                        print(f"Length scale set to {length_scale}")
                    except ValueError:
                        print("Invalid length scale")
                    continue
                
                elif user_input.lower().startswith('t '):
                    try:
                        temperature = float(user_input[2:])
                        print(f"Temperature set to {temperature}")
                    except ValueError:
                        print("Invalid temperature")
                    continue
                
                # Synthesize text
                try:
                    counter += 1
                    save_path = os.path.join(output_dir, f"output_{counter:03d}.wav")
                    
                    print(f"Synthesizing: {user_input}")
                    waveform = self.inference_engine.synthesize(
                        kannada_text=user_input,
                        temperature=temperature,
                        length_scale=length_scale,
                        pitch=pitch,
                        energy=energy,
                        save_path=save_path
                    )
                    print(f"✓ Synthesis complete! Saved to {save_path}\n")
                
                except Exception as e:
                    print(f"✗ Synthesis failed: {str(e)}\n")
            
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='HKL-VITS Inference')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/hkl_vits_config.json',
        help='Path to config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--text',
        type=str,
        default=None,
        help='Input text (if not provided, run interactive mode)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output.wav',
        help='Output audio file path'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.667,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--length_scale',
        type=float,
        default=1.0,
        help='Duration scale factor'
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    
    if args.interactive:
        # Interactive mode
        interactive = InteractiveInference(args.config, args.checkpoint, device=device)
        interactive.run()
    elif args.text:
        # Single text synthesis
        inference = HKLVITSInference(args.config, args.checkpoint, device=device)
        waveform = inference.synthesize(
            kannada_text=args.text,
            temperature=args.temperature,
            length_scale=args.length_scale,
            save_path=args.output
        )
        print(f"Synthesis complete! Audio saved to {args.output}")
    else:
        print("Please provide either --text or --interactive flag")


if __name__ == '__main__':
    main()
