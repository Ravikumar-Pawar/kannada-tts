#!/usr/bin/env python3
"""
Kannada TTS Utilities - Helper functions and scripts
"""

import os
import json
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. MODEL UTILITIES
# ============================================================================
class ModelUtils:
    """Model management utilities"""
    
    @staticmethod
    def get_model_size(model_path: str) -> Dict[str, float]:
        """Get model size information"""
        if not os.path.exists(model_path):
            return {"error": "Model not found"}
        
        size_bytes = os.path.getsize(model_path)
        size_mb = size_bytes / (1024 ** 2)
        size_gb = size_bytes / (1024 ** 3)
        
        return {
            "bytes": size_bytes,
            "MB": round(size_mb, 2),
            "GB": round(size_gb, 2),
            "path": model_path
        }
    
    @staticmethod
    def list_checkpoints(checkpoint_dir: str) -> List[Dict]:
        """List all model checkpoints"""
        if not os.path.exists(checkpoint_dir):
            return []
        
        checkpoints = []
        for file in sorted(Path(checkpoint_dir).glob("checkpoint_*.pth")):
            size_mb = os.path.getsize(file) / (1024 ** 2)
            checkpoints.append({
                "filename": file.name,
                "path": str(file),
                "size_mb": round(size_mb, 2),
                "modified": os.path.getmtime(file)
            })
        
        return sorted(checkpoints, key=lambda x: x["modified"], reverse=True)
    
    @staticmethod
    def estimate_inference_time(audio_duration_seconds: float, model: str = "tacotron2") -> Dict:
        """Estimate inference time based on audio duration"""
        
        # Empirical values
        timings = {
            "tacotron2": {
                "cpu": 3.5,  # RTF (Real-Time Factor)
                "gpu": 0.2
            },
            "hifigan": {
                "cpu": 2.0,
                "gpu": 0.05
            }
        }
        
        device = "gpu" if torch.cuda.is_available() else "cpu"
        rtf = timings.get(model, {}).get(device, 1.0)
        
        estimated_time = audio_duration_seconds * rtf
        
        return {
            "audio_duration_sec": audio_duration_seconds,
            "device": device,
            "model": model,
            "rtf": rtf,
            "estimated_time_sec": round(estimated_time, 2),
            "estimated_time_min": round(estimated_time / 60, 2)
        }

# ============================================================================
# 2. DATASET UTILITIES
# ============================================================================
class DatasetUtils:
    """Dataset analysis and manipulation"""
    
    @staticmethod
    def load_metadata(csv_path: str) -> pd.DataFrame:
        """Load metadata CSV"""
        try:
            if csv_path.endswith('_extended.csv'):
                return pd.read_csv(csv_path)
            else:
                return pd.read_csv(csv_path, sep='|', header=None, names=['wav_path', 'text'])
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def analyze_dataset(metadata_path: str) -> Dict:
        """Analyze dataset statistics"""
        df = DatasetUtils.load_metadata(metadata_path)
        
        if df.empty:
            return {"error": "Empty dataset"}
        
        stats = {
            "total_samples": len(df),
            "text_stats": {
                "min_length": int(df['text'].str.len().min()),
                "max_length": int(df['text'].str.len().max()),
                "mean_length": int(df['text'].str.len().mean()),
                "median_length": int(df['text'].str.len().median())
            }
        }
        
        # If extended metadata
        if 'duration' in df.columns:
            stats.update({
                "duration_stats": {
                    "min_sec": round(df['duration'].min(), 2),
                    "max_sec": round(df['duration'].max(), 2),
                    "mean_sec": round(df['duration'].mean(), 2),
                    "total_hours": round(df['duration'].sum() / 3600, 2)
                },
                "rms_energy_stats": {
                    "min": round(df['rms_energy'].min(), 4),
                    "max": round(df['rms_energy'].max(), 4),
                    "mean": round(df['rms_energy'].mean(), 4)
                }
            })
        
        return stats
    
    @staticmethod
    def sample_random_texts(metadata_path: str, n_samples: int = 10) -> List[str]:
        """Get random Kannada text samples"""
        df = DatasetUtils.load_metadata(metadata_path)
        if df.empty:
            return []
        
        samples = df['text'].sample(n=min(n_samples, len(df))).tolist()
        return samples

# ============================================================================
# 3. AUDIO UTILITIES
# ============================================================================
class AudioUtils:
    """Audio processing utilities"""
    
    @staticmethod
    def load_audio(audio_path: str, sr: int = 22050) -> np.ndarray:
        """Load audio file"""
        try:
            y, _ = librosa.load(audio_path, sr=sr)
            return y
        except Exception as e:
            print(f"Error loading audio: {e}")
            return np.array([])
    
    @staticmethod
    def get_audio_info(audio_path: str) -> Dict:
        """Get audio file information"""
        try:
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            return {
                "path": audio_path,
                "sample_rate": sr,
                "duration_sec": round(duration, 2),
                "samples": len(y),
                "rms_energy": round(np.sqrt(np.mean(y**2)), 4),
                "peak_value": round(np.max(np.abs(y)), 4),
                "file_size_mb": round(os.path.getsize(audio_path) / (1024**2), 2)
            }
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def plot_waveform(audio_path: str, title: str = "Waveform", save_path: str = None):
        """Plot waveform"""
        try:
            y, sr = librosa.load(audio_path, sr=22050)
            
            plt.figure(figsize=(14, 4))
            librosa.display.waveshow(y, sr=sr, alpha=0.8)
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.show()
        
        except Exception as e:
            print(f"Error plotting waveform: {e}")
    
    @staticmethod
    def plot_spectrogram(audio_path: str, title: str = "Mel-Spectrogram", save_path: str = None):
        """Plot mel-spectrogram"""
        try:
            y, sr = librosa.load(audio_path, sr=22050)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            S_db = librosa.power_to_db(S, ref=np.max)
            
            plt.figure(figsize=(14, 5))
            img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
            plt.colorbar(img, format='%+2.0f dB')
            plt.title(title, fontsize=14, fontweight='bold')
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.show()
        
        except Exception as e:
            print(f"Error plotting spectrogram: {e}")

# ============================================================================
# 4. RESULTS UTILITIES
# ============================================================================
class ResultsUtils:
    """Results analysis and reporting"""
    
    @staticmethod
    def load_evaluation_results(json_path: str) -> Dict:
        """Load evaluation results"""
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading results: {e}")
            return {}
    
    @staticmethod
    def generate_report(metadata_path: str, eval_results_path: str = None) -> str:
        """Generate comprehensive report"""
        report = []
        report.append("="*70)
        report.append("KANNADA TTS SYSTEM REPORT")
        report.append("="*70)
        report.append("")
        
        # Dataset info
        dataset_stats = DatasetUtils.analyze_dataset(metadata_path)
        report.append("📊 DATASET ANALYSIS")
        report.append("-" * 70)
        for key, value in dataset_stats.items():
            report.append(f"{key}: {value}")
        report.append("")
        
        # Evaluation results (if available)
        if eval_results_path and os.path.exists(eval_results_path):
            eval_results = ResultsUtils.load_evaluation_results(eval_results_path)
            report.append("📈 EVALUATION METRICS")
            report.append("-" * 70)
            if "summary" in eval_results:
                for metric, value in eval_results["summary"].items():
                    if isinstance(value, float):
                        report.append(f"{metric}: {value:.4f}")
                    else:
                        report.append(f"{metric}: {value}")
            report.append("")
        
        report.append("="*70)
        return "\n".join(report)
    
    @staticmethod
    def print_report(metadata_path: str, eval_results_path: str = None):
        """Print comprehensive report"""
        report = ResultsUtils.generate_report(metadata_path, eval_results_path)
        print(report)
        
        # Save to file
        with open("output/system_report.txt", "w") as f:
            f.write(report)
        print("\n✅ Report saved to: output/system_report.txt")

# ============================================================================
# 5. SYSTEM UTILITIES
# ============================================================================
class SystemUtils:
    """System information and diagnostics"""
    
    @staticmethod
    def get_system_info() -> Dict:
        """Get system and GPU information"""
        info = {
            "device": "GPU" if torch.cuda.is_available() else "CPU",
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info.update({
                "gpu_device": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
                "cuda_version": torch.version.cuda,
                "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            })
        
        return info
    
    @staticmethod
    def check_disk_space() -> Dict:
        """Check available disk space"""
        import shutil
        
        stat = shutil.disk_usage(".")
        
        return {
            "total_gb": round(stat.total / (1024**3), 2),
            "used_gb": round(stat.used / (1024**3), 2),
            "free_gb": round(stat.free / (1024**3), 2),
            "percent_used": round((stat.used / stat.total) * 100, 1)
        }
    
    @staticmethod
    def print_diagnostics():
        """Print system diagnostics"""
        print("\n" + "="*70)
        print("🔧 SYSTEM DIAGNOSTICS")
        print("="*70)
        
        # System info
        print("\n🖥️  HARDWARE:")
        sys_info = SystemUtils.get_system_info()
        for key, value in sys_info.items():
            print(f"  {key}: {value}")
        
        # Disk space
        print("\n💾 STORAGE:")
        disk_info = SystemUtils.check_disk_space()
        for key, value in disk_info.items():
            print(f"  {key}: {value}")
        
        print("\n" + "="*70)

# ============================================================================
# 6. MAIN UTILITY RUNNER
# ============================================================================
if __name__ == "__main__":
    print("Kannada TTS - Utility Module")
    print("\nUsage Examples:")
    print("  from src.utils import *")
    print("  DatasetUtils.analyze_dataset('data/metadata.csv')")
    print("  AudioUtils.get_audio_info('output/inference/test.wav')")
    print("  SystemUtils.print_diagnostics()")
    
    # Demo
    print("\n" + "="*70)
    print("RUNNING DIAGNOSTICS")
    print("="*70)
    SystemUtils.print_diagnostics()
