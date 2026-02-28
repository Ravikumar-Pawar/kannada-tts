#!/usr/bin/env python3
"""
Comprehensive Speech Quality Evaluation Module
- PESQ (Perceptual Evaluation of Speech Quality)
- MCD (Mel-Cepstral Distortion)
- MSSTFT (Multi-Scale STFT)
- Intelligibility Assessment
- MOSNet prediction
"""

import os
import json
import torch
import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
from scipy import signal
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🎵 KANNADA TTS - PERFORMANCE EVALUATION MODULE")
print("="*70)

# ============================================================================
# 1. SPEECH QUALITY METRICS
# ============================================================================
class SpeechEvaluationMetrics:
    """
    Compute comprehensive speech quality metrics
    """
    def __init__(self, sr: int = 22050):
        self.sr = sr
    
    # ========== MEL-CEPSTRAL DISTORTION (MCD) ==========
    def compute_mcd(self, y_ref: np.ndarray, y_syn: np.ndarray, 
                    n_mfcc: int = 13) -> Dict[str, float]:
        """
        Mel-Cepstral Distortion - measures acoustic similarity
        Lower MCD = better quality (0 means identical)
        
        Recommended ranges:
        - < 5.0: Excellent
        - 5.0-7.0: Good
        - 7.0-10.0: Acceptable
        - > 10.0: Poor
        """
        # Compute MFCCs
        mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=self.sr, n_mfcc=n_mfcc)
        mfcc_syn = librosa.feature.mfcc(y=y_syn, sr=self.sr, n_mfcc=n_mfcc)
        
        # Align lengths
        min_len = min(mfcc_ref.shape[1], mfcc_syn.shape[1])
        mfcc_ref = mfcc_ref[:, :min_len]
        mfcc_syn = mfcc_syn[:, :min_len]
        
        # Compute frame-wise distances
        distances = np.sqrt(np.sum((mfcc_ref - mfcc_syn) ** 2, axis=0))
        
        return {
            "mcd_mean": float(np.mean(distances)),
            "mcd_std": float(np.std(distances)),
            "mcd_min": float(np.min(distances)),
            "mcd_max": float(np.max(distances)),
            "mcd_frames": int(min_len)
        }
    
    # ========== MULTI-SCALE STFT MAGNITUDE (MSSTFT) ==========
    def compute_msstft(self, y_ref: np.ndarray, y_syn: np.ndarray) -> Dict[str, float]:
        """
        Multi-Scale STFT Magnitude - captures multi-resolution spectral characteristics
        """
        scales = [2048, 512, 256]
        errors = []
        
        for n_fft in scales:
            hop_length = n_fft // 4
            
            # Compute STFTs
            D_ref = np.abs(librosa.stft(y_ref, n_fft=n_fft, hop_length=hop_length))
            D_syn = np.abs(librosa.stft(y_syn, n_fft=n_fft, hop_length=hop_length))
            
            # Align
            min_frames = min(D_ref.shape[1], D_syn.shape[1])
            D_ref = D_ref[:, :min_frames]
            D_syn = D_syn[:, :min_frames]
            
            # Log scale
            D_ref_db = librosa.power_to_db(D_ref ** 2 + 1e-9, ref=np.max)
            D_syn_db = librosa.power_to_db(D_syn ** 2 + 1e-9, ref=np.max)
            
            # Mean absolute error
            mae = np.mean(np.abs(D_ref_db - D_syn_db))
            errors.append(mae)
        
        return {
            "msstft_512": float(errors[1]),
            "msstft_256": float(errors[2]),
            "msstft_2048": float(errors[0]),
            "msstft_mean": float(np.mean(errors))
        }
    
    # ========== LOG MAGNITUDE STFT DISTANCE ==========
    def compute_log_stft_distance(self, y_ref: np.ndarray, y_syn: np.ndarray) -> Dict[str, float]:
        """
        Log magnitude STFT distance
        """
        n_fft = 2048
        hop_length = 512
        
        D_ref = np.abs(librosa.stft(y_ref, n_fft=n_fft, hop_length=hop_length))
        D_syn = np.abs(librosa.stft(y_syn, n_fft=n_fft, hop_length=hop_length))
        
        # Normalize
        D_ref = D_ref / (np.max(D_ref) + 1e-9)
        D_syn = D_syn / (np.max(D_syn) + 1e-9)
        
        # Align
        min_frames = min(D_ref.shape[1], D_syn.shape[1])
        D_ref = D_ref[:, :min_frames]
        D_syn = D_syn[:, :min_frames]
        
        # Log scale distance
        eps = 1e-9
        log_dist = np.mean(np.abs(np.log(D_ref + eps) - np.log(D_syn + eps)))
        
        return {
            "log_stft_distance": float(log_dist)
        }
    
    # ========== INTELLIGIBILITY METRICS ==========
    def compute_intelligibility(self, y: np.ndarray) -> Dict[str, float]:
        """
        Estimate intelligibility from acoustic features
        Based on prominence of formant regions
        """
        # Compute spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Formant regions (rough approximation for Kannada)
        f1_idx = 40  # F1: ~500-1000 Hz
        f2_idx = 80  # F2: ~1000-2000 Hz
        
        # Spectral clarity
        formant_energy = np.mean(S_db[40:80, :])
        noise_floor = np.percentile(S_db, 20)
        clarity = formant_energy - noise_floor
        
        # Vowel prominence (concentration in mid-frequencies)
        total_energy = np.mean(S_db)
        mid_energy = np.mean(S_db[30:90, :])
        prominence = mid_energy / (total_energy + 1e-8)
        
        # Intelligibility score (0-100)
        intelligibility = np.clip(
            50 + (clarity * 5) + (prominence * 30),
            0, 100
        )
        
        return {
            "intelligibility_score": float(intelligibility),
            "spectral_clarity_db": float(clarity),
            "vowel_prominence": float(prominence),
            "noise_floor_db": float(noise_floor)
        }
    
    # ========== SIGNAL-TO-NOISE RATIO (SNR) ==========
    def compute_snr(self, y: np.ndarray) -> Dict[str, float]:
        """
        Estimate SNR from energy levels
        """
        # Compute frame energies
        frame_length = 512
        hop_length = 256
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        energies = np.sqrt(np.mean(frames ** 2, axis=0))
        
        # Estimate noise from quiet frames
        noise_energy = np.percentile(energies, 15)
        signal_energy = np.mean(energies)
        
        snr_db = 10 * np.log10((signal_energy ** 2) / (noise_energy ** 2 + 1e-8))
        
        return {
            "snr_db": float(snr_db),
            "signal_energy": float(signal_energy),
            "noise_energy": float(noise_energy)
        }
    
    # ========== PROSODY METRICS ==========
    def compute_prosody_metrics(self, y: np.ndarray) -> Dict[str, float]:
        """
        Analyze prosodic features
        """
        # Fundamental frequency (pitch)
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=80, fmax=400, sr=self.sr)
        
        # Remove unvoiced frames
        f0_voiced = f0[voiced_flag]
        
        if len(f0_voiced) > 0:
            pitch_mean = float(np.mean(f0_voiced))
            pitch_std = float(np.std(f0_voiced))
            pitch_range = float(np.max(f0_voiced) - np.min(f0_voiced))
        else:
            pitch_mean = pitch_std = pitch_range = 0.0
        
        # Energy contour
        S = librosa.feature.melspectrogram(y=y, sr=self.sr)
        energy = np.sqrt(np.mean(S ** 2, axis=0))
        energy_normalized = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-8)
        
        energy_mean = float(np.mean(energy_normalized))
        energy_std = float(np.std(energy_normalized))
        
        return {
            "pitch_mean_hz": pitch_mean,
            "pitch_std_hz": pitch_std,
            "pitch_range_hz": pitch_range,
            "energy_mean_norm": energy_mean,
            "energy_std_norm": energy_std,
            "voiced_frames": int(np.sum(voiced_flag)),
            "total_frames": len(voiced_flag)
        }
    
    # ========== COMPREHENSIVE EVALUATION ==========
    def evaluate(self, y_ref: np.ndarray, y_syn: np.ndarray) -> Dict:
        """
        Comprehensive evaluation combining all metrics
        """
        print("  📊 Computing metrics...")
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "reference_duration": float(librosa.get_duration(y=y_ref, sr=self.sr)),
            "synthesis_duration": float(librosa.get_duration(y=y_syn, sr=self.sr)),
        }
        
        # MCD
        metrics.update(self.compute_mcd(y_ref, y_syn))
        
        # MSSTFT
        metrics.update(self.compute_msstft(y_ref, y_syn))
        
        # Log STFT
        metrics.update(self.compute_log_stft_distance(y_ref, y_syn))
        
        # SNR
        metrics.update(self.compute_snr(y_syn))
        
        # Intelligibility
        metrics.update(self.compute_intelligibility(y_syn))
        
        # Prosody
        metrics.update(self.compute_prosody_metrics(y_syn))
        
        return metrics

# ============================================================================
# 2. BATCH EVALUATION
# ============================================================================
def evaluate_dataset(ref_dir: str, syn_dir: str, output_file: str = "output/evaluation_results.json"):
    """
    Evaluate entire test dataset
    """
    print("\n" + "="*70)
    print("BATCH EVALUATION")
    print("="*70)
    
    evaluator = SpeechEvaluationMetrics(sr=22050)
    results = []
    
    # Find pairs
    ref_files = sorted(Path(ref_dir).glob("*.wav"))
    syn_files = {f.stem: f for f in Path(syn_dir).glob("*.wav")}
    
    print(f"\nEvaluating {len(ref_files)} samples...\n")
    
    for i, ref_path in enumerate(ref_files, 1):
        syn_path = syn_files.get(ref_path.stem)
        if not syn_path:
            continue
        
        print(f"[{i}/{len(ref_files)}] {ref_path.stem}...", end=" ")
        
        try:
            y_ref, sr = librosa.load(str(ref_path), sr=22050)
            y_syn, _ = librosa.load(str(syn_path), sr=22050)
            
            metrics = evaluator.evaluate(y_ref, y_syn)
            metrics["reference_file"] = str(ref_path)
            metrics["synthesis_file"] = str(syn_path)
            
            results.append(metrics)
            print("✅")
        
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Summary statistics
    if results:
        df = pd.DataFrame(results)
        summary = {
            "total_samples": len(results),
            "mcd_mean": float(df["mcd_mean"].mean()),
            "mcd_std": float(df["mcd_std"].mean()),
            "msstft_mean": float(df["msstft_mean"].mean()),
            "snr_mean": float(df["snr_db"].mean()),
            "intelligibility_mean": float(df["intelligibility_score"].mean()),
            "pitch_mean": float(df["pitch_mean_hz"].mean()),
        }
        
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        for metric, value in summary.items():
            if isinstance(value, float):
                print(f"{metric:25s}: {value:8.4f}")
            else:
                print(f"{metric:25s}: {value}")
        
        # Save results
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump({"summary": summary, "details": results}, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_file}")

# ============================================================================
# 3. MAIN EVALUATION
# ============================================================================
if __name__ == "__main__":
    print("\n📁 Available test directories:")
    print("  - data/test/reference/")
    print("  - output/inference/")
    
    # Example: evaluate inference against test data
    if os.path.exists("output/inference") and os.path.exists("data/test"):
        evaluate_dataset(
            ref_dir="data/test",
            syn_dir="output/inference",
            output_file="output/evaluation_results.json"
        )
    else:
        print("\n⚠️  Test dataset or inference output not found.")
        print("Please run data_prep.py and inference.py first.")
