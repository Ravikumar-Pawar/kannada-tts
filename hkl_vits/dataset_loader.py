# dataset_loader.py

import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional
import json
import librosa
from pathlib import Path

from kannada_g2p import KannadaG2P


class KannadaTTSDataset(Dataset):
    """
    Dataset loader for Kannada Text-to-Speech
    Loads audio files and corresponding text transcriptions
    Extracts prosody features (pitch and energy)
    """
    
    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        f0_min: float = 70,
        f0_max: float = 400,
        trim_silence: bool = True,
        max_audio_length: int = 50000,
        min_audio_length: int = 1000
    ):
        """
        Args:
            data_dir: Path to dataset directory containing 'wav' and 'txt' folders
            sample_rate: Audio sample rate
            n_fft: FFT size for mel-spectrogram
            hop_length: Hop length for STFT
            n_mels: Number of mel frequency bins
            f0_min: Minimum F0 frequency for pitch extraction
            f0_max: Maximum F0 frequency for pitch extraction
            trim_silence: Whether to trim silence from audio
            max_audio_length: Maximum audio length in samples
            min_audio_length: Minimum audio length in samples
        """
        self.data_dir = Path(data_dir)
        self.wav_dir = self.data_dir / 'wav'
        self.txt_dir = self.data_dir / 'txt'
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.trim_silence = trim_silence
        self.max_audio_length = max_audio_length
        self.min_audio_length = min_audio_length
        
        # Initialize G2P converter
        self.g2p = KannadaG2P()
        
        # Build file list
        self.file_list = self._build_file_list()
        
        # Mel spectrogram transform
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
        print(f"Loaded {len(self.file_list)} samples from {data_dir}")

    def _build_file_list(self) -> list:
        """Build list of (wav_file, txt_file) pairs"""
        file_list = []
        
        if not self.wav_dir.exists() or not self.txt_dir.exists():
            print(f"Warning: Data directories not found at {self.data_dir}")
            return file_list
        
        wav_files = sorted(self.wav_dir.glob('*.wav'))
        
        for wav_file in wav_files:
            txt_file = self.txt_dir / (wav_file.stem + '.txt')
            if txt_file.exists():
                file_list.append((wav_file, txt_file))
        
        return file_list

    def _read_text(self, txt_file: Path) -> str:
        """Read text from file"""
        with open(txt_file, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def _load_audio(self, wav_file: Path) -> torch.Tensor:
        """Load and process audio file"""
        waveform, sr = torchaudio.load(str(wav_file))
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Trim silence
        if self.trim_silence:
            waveform, _ = torchaudio.functional.vad(waveform, self.sample_rate)
        
        # Check length constraints
        if waveform.shape[1] < self.min_audio_length:
            return None
        
        if waveform.shape[1] > self.max_audio_length:
            waveform = waveform[:, :self.max_audio_length]
        
        return waveform.squeeze(0)

    def _extract_pitch(self, waveform: torch.Tensor) -> np.ndarray:
        """Extract fundamental frequency (F0) using librosa"""
        audio = waveform.numpy()
        
        # Extract pitch using pYIN algorithm
        f0 = librosa.yin(
            audio,
            fmin=self.f0_min,
            fmax=self.f0_max,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        return f0

    def _extract_energy(self, waveform: torch.Tensor) -> np.ndarray:
        """Extract energy from audio"""
        # Compute Short-Time Fourier Transform
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True
        )
        
        # Compute magnitude
        magnitude = torch.abs(stft)
        
        # Sum magnitude over frequency bins to get energy
        energy = magnitude.sum(dim=0).numpy()
        
        # Normalize energy
        energy = (energy - energy.mean()) / (energy.std() + 1e-6)
        
        return energy

    def _get_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute mel-spectrogram"""
        mel_spec = self.mel_spectrogram(waveform)
        # Convert to log scale
        mel_spec_db = torch.log(torch.clamp(mel_spec, min=1e-9))
        return mel_spec_db

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset sample
        
        Returns:
            Dictionary containing:
            - text: grapheme IDs
            - phonemes: phoneme IDs
            - audio: waveform
            - mel_spec: mel-spectrogram
            - pitch: fundamental frequency
            - energy: energy contour
            - text_len: length of text
            - audio_len: length of audio
        """
        wav_file, txt_file = self.file_list[idx]
        
        # Load text
        text = self._read_text(txt_file)
        
        # Load audio
        waveform = self._load_audio(wav_file)
        if waveform is None:
            # Return empty sample if audio is invalid
            return self.__getitem__((idx + 1) % len(self))
        
        # Convert text to grapheme and phoneme IDs
        grapheme_ids = torch.tensor(
            [ord(c) for c in text if ord(c) < 3000],  # Filter Kannada characters
            dtype=torch.long
        )
        phoneme_ids, phoneme_lengths = self.g2p.batch_text_to_phoneme_ids([text])
        phoneme_ids = phoneme_ids[0]
        
        # Extract prosody features
        pitch = self._extract_pitch(waveform)
        energy = self._extract_energy(waveform)
        
        # Compute mel-spectrogram
        mel_spec = self._get_mel_spectrogram(waveform)
        
        # Align pitch and energy to mel-spectrogram length
        mel_len = mel_spec.shape[1]
        
        # Interpolate pitch and energy to mel-spectrogram length
        if len(pitch) != mel_len:
            pitch = np.interp(
                np.linspace(0, 1, mel_len),
                np.linspace(0, 1, len(pitch)),
                pitch
            )
        
        if len(energy) != mel_len:
            energy = np.interp(
                np.linspace(0, 1, mel_len),
                np.linspace(0, 1, len(energy)),
                energy
            )
        
        pitch = torch.from_numpy(pitch).float()
        energy = torch.from_numpy(energy).float()
        
        return {
            'text': grapheme_ids,
            'phonemes': phoneme_ids,
            'audio': waveform,
            'mel_spec': mel_spec,
            'pitch': pitch,
            'energy': energy,
            'text_len': len(grapheme_ids),
            'audio_len': len(waveform),
            'filename': str(wav_file.name)
        }


def collate_fn(batch):
    """
    Custom collate function for batching samples
    Pads sequences to the same length
    """
    # Filter out None samples
    batch = [s for s in batch if s is not None]
    
    if len(batch) == 0:
        return None
    
    # Pad text
    text_lengths = [s['text_len'] for s in batch]
    max_text_len = max(text_lengths)
    texts = torch.zeros(len(batch), max_text_len, dtype=torch.long)
    for i, sample in enumerate(batch):
        text = sample['text']
        texts[i, :len(text)] = text
    
    # Pad phonemes
    phoneme_lengths = torch.tensor(
        [s['phonemes'].shape[0] for s in batch],
        dtype=torch.long
    )
    max_phoneme_len = phoneme_lengths.max().item()
    phonemes = torch.zeros(len(batch), max_phoneme_len, dtype=torch.long)
    for i, sample in enumerate(batch):
        phoneme = sample['phonemes']
        phonemes[i, :len(phoneme)] = phoneme
    
    # Pad audio
    audio_lengths = [s['audio_len'] for s in batch]
    max_audio_len = max(audio_lengths)
    audios = torch.zeros(len(batch), max_audio_len)
    for i, sample in enumerate(batch):
        audio = sample['audio']
        audios[i, :len(audio)] = audio
    
    # Pad mel-spectrograms
    mel_lengths = [s['mel_spec'].shape[1] for s in batch]
    max_mel_len = max(mel_lengths)
    mel_specs = torch.zeros(len(batch), batch[0]['mel_spec'].shape[0], max_mel_len)
    for i, sample in enumerate(batch):
        mel_spec = sample['mel_spec']
        mel_specs[i, :, :mel_spec.shape[1]] = mel_spec
    
    # Pad pitch and energy
    max_prosody_len = max_mel_len
    pitches = torch.zeros(len(batch), max_prosody_len)
    energies = torch.zeros(len(batch), max_prosody_len)
    for i, sample in enumerate(batch):
        pitch = sample['pitch']
        energy = sample['energy']
        pitches[i, :len(pitch)] = pitch
        energies[i, :len(energy)] = energy
    
    return {
        'text': texts,
        'text_lengths': torch.tensor(text_lengths, dtype=torch.long),
        'phonemes': phonemes,
        'phoneme_lengths': phoneme_lengths,
        'audio': audios,
        'audio_lengths': torch.tensor(audio_lengths, dtype=torch.long),
        'mel_spec': mel_specs,
        'mel_lengths': torch.tensor(mel_lengths, dtype=torch.long),
        'pitch': pitches.unsqueeze(2),  # (batch, time, 1)
        'energy': energies.unsqueeze(2),  # (batch, time, 1)
    }


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: float = 0.8,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        train_split: Fraction of data for training
        **kwargs: Additional arguments for KannadaTTSDataset
    
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    dataset = KannadaTTSDataset(data_dir, **kwargs)
    
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader
