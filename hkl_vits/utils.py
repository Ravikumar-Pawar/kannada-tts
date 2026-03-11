# utils.py

import os
import json
import argparse
from pathlib import Path
import torch
import torchaudio
import librosa
import numpy as np
from typing import Tuple, List
import logging

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def setup_logger(name: str = '__main__') -> logging.Logger:
    """Setup logging"""
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def validate_kannada_text(text: str) -> bool:
    """
    Validate if text contains Kannada characters
    
    Args:
        text: Text to validate
    
    Returns:
        True if text contains Kannada characters
    """
    kannada_range = range(0x0C80, 0x0CFF)  # Kannada Unicode range
    return any(ord(c) in kannada_range for c in text)


def prepare_dataset(data_dir: str, output_dir: str = None):
    """
    Prepare dataset for training
    Validates audio-text pairs and preprocessing
    
    Args:
        data_dir: Path to dataset directory
        output_dir: Path to save preprocessed data (optional)
    """
    logger = setup_logger('dataset_prep')
    
    data_path = Path(data_dir)
    wav_dir = data_path / 'wav'
    txt_dir = data_path / 'txt'
    
    if not wav_dir.exists() or not txt_dir.exists():
        logger.error(f"Dataset directories not found in {data_dir}")
        return
    
    # Collect valid samples
    valid_samples = []
    invalid_samples = []
    
    wav_files = sorted(wav_dir.glob('*.wav'))
    logger.info(f"Found {len(wav_files)} audio files")
    
    for i, wav_file in enumerate(wav_files):
        if (i + 1) % 100 == 0:
            logger.info(f"Processing {i + 1}/{len(wav_files)}")
        
        txt_file = txt_dir / (wav_file.stem + '.txt')
        
        # Check if text file exists
        if not txt_file.exists():
            invalid_samples.append((wav_file.name, "No text file"))
            continue
        
        # Read text
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        except Exception as e:
            invalid_samples.append((wav_file.name, f"Text read error: {str(e)}"))
            continue
        
        # Validate text
        if not text:
            invalid_samples.append((wav_file.name, "Empty text"))
            continue
        
        if not validate_kannada_text(text):
            invalid_samples.append((wav_file.name, "Not Kannada text"))
            continue
        
        # Load audio and validate
        try:
            waveform, sr = torchaudio.load(str(wav_file))
            duration = waveform.shape[1] / sr
            
            # Check duration (1-10 seconds recommended)
            if duration < 0.5:
                invalid_samples.append((wav_file.name, "Audio too short"))
                continue
            if duration > 30:
                invalid_samples.append((wav_file.name, "Audio too long"))
                continue
            
            # Check audio quality
            rms = torch.sqrt(torch.mean(waveform ** 2))
            if rms < 0.001:
                invalid_samples.append((wav_file.name, "Audio too quiet"))
                continue
            
            valid_samples.append((wav_file.name, text, duration))
        
        except Exception as e:
            invalid_samples.append((wav_file.name, f"Audio error: {str(e)}"))
            continue
    
    # Print statistics
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset Preparation Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Total samples: {len(wav_files)}")
    logger.info(f"Valid samples: {len(valid_samples)}")
    logger.info(f"Invalid samples: {len(invalid_samples)}")
    
    if valid_samples:
        durations = [d for _, _, d in valid_samples]
        logger.info(f"Average duration: {np.mean(durations):.2f}s")
        logger.info(f"Min duration: {min(durations):.2f}s")
        logger.info(f"Max duration: {max(durations):.2f}s")
    
    if invalid_samples:
        logger.info(f"\nInvalid samples:")
        for filename, reason in invalid_samples[:10]:
            logger.info(f"  - {filename}: {reason}")
        if len(invalid_samples) > 10:
            logger.info(f"  ... and {len(invalid_samples) - 10} more")
    
    # Save valid samples list
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        stats = {
            'total_samples': len(wav_files),
            'valid_samples': len(valid_samples),
            'invalid_samples': len(invalid_samples),
            'valid_files': [f for f, _, _ in valid_samples],
            'statistics': {
                'avg_duration': float(np.mean([d for _, _, d in valid_samples])) if valid_samples else 0,
                'total_duration': float(sum(d for _, _, d in valid_samples))
            }
        }
        
        stats_file = output_path / 'dataset_stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nDataset statistics saved to {stats_file}")


def compute_mel_spectrogram(
    waveform: torch.Tensor,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
    sample_rate: int = 22050
) -> np.ndarray:
    """Compute mel-spectrogram"""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    mel_spec = mel_transform(waveform)
    mel_spec_db = torch.log(torch.clamp(mel_spec, min=1e-9))
    return mel_spec_db.numpy()


def extract_f0(
    waveform: torch.Tensor,
    sample_rate: int = 22050,
    f0_min: float = 70,
    f0_max: float = 400,
    hop_length: int = 256
) -> np.ndarray:
    """Extract fundamental frequency (F0) using PYIN"""
    audio = waveform.numpy()
    f0 = librosa.yin(
        audio,
        fmin=f0_min,
        fmax=f0_max,
        sr=sample_rate,
        hop_length=hop_length
    )
    return f0


def extract_energy(
    waveform: torch.Tensor,
    n_fft: int = 1024,
    hop_length: int = 256
) -> np.ndarray:
    """Extract energy contour"""
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True
    )
    magnitude = torch.abs(stft)
    energy = magnitude.sum(dim=0).numpy()
    energy = (energy - energy.mean()) / (energy.std() + 1e-6)
    return energy


def normalize_waveform(waveform: torch.Tensor, target_db: float = -20) -> torch.Tensor:
    """Normalize waveform to target loudness"""
    rms = torch.sqrt(torch.mean(waveform ** 2))
    current_db = 20 * np.log10(rms + 1e-10)
    target_linear = 10 ** (target_db / 20)
    normalized = waveform * (target_linear / (rms + 1e-10))
    return torch.clamp(normalized, -1, 1)


def split_dataset(
    data_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    output_dir: str = None
):
    """
    Split dataset into train/val/test sets
    
    Args:
        data_dir: Path to dataset
        train_ratio: Ratio for training (default 0.8)
        val_ratio: Ratio for validation (default 0.1)
        output_dir: Output directory for split info
    """
    logger = setup_logger('dataset_split')
    
    data_path = Path(data_dir)
    wav_dir = data_path / 'wav'
    
    wav_files = sorted(wav_dir.glob('*.wav'))
    logger.info(f"Total samples: {len(wav_files)}")
    
    # Compute split indices
    n_samples = len(wav_files)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val
    
    # Random shuffle
    indices = np.random.permutation(n_samples)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Create split dictionaries
    splits = {
        'train': [wav_files[i].stem for i in train_indices],
        'val': [wav_files[i].stem for i in val_indices],
        'test': [wav_files[i].stem for i in test_indices]
    }
    
    logger.info(f"Train: {len(splits['train'])}")
    logger.info(f"Val: {len(splits['val'])}")
    logger.info(f"Test: {len(splits['test'])}")
    
    # Save split info
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        split_file = output_path / 'dataset_split.json'
        with open(split_file, 'w') as f:
            json.dump(splits, f, indent=2)
        
        logger.info(f"Split info saved to {split_file}")


def print_config(config_path: str):
    """Print configuration"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(json.dumps(config, indent=2))


def main():
    parser = argparse.ArgumentParser(description='HKL-VITS Utilities')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Prepare dataset command
    prepare_parser = subparsers.add_parser('prepare_dataset', help='Prepare dataset')
    prepare_parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    prepare_parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    
    # Split dataset command
    split_parser = subparsers.add_parser('split_dataset', help='Split dataset')
    split_parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    split_parser.add_argument('--train_ratio', type=float, default=0.8, help='Training ratio')
    split_parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio')
    split_parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    
    # Print config command
    config_parser = subparsers.add_parser('print_config', help='Print configuration')
    config_parser.add_argument('--config', type=str, required=True, help='Config file')
    
    args = parser.parse_args()
    
    if args.command == 'prepare_dataset':
        prepare_dataset(args.data_dir, args.output_dir)
    
    elif args.command == 'split_dataset':
        split_dataset(args.data_dir, args.train_ratio, args.val_ratio, args.output_dir)
    
    elif args.command == 'print_config':
        print_config(args.config)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
