#!/usr/bin/env python3
"""
Advanced Kannada TTS Inference with Noise Reduction and Emotion Enhancement
- Tacotron2 acoustic model synthesis
- HiFiGAN vocoder
- Noise reduction (spectral gating)
- Emotion/prosody enhancement
- Speech quality assessment
"""

import os
import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from datetime import datetime
from scipy import signal
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🎵 KANNADA TTS - ADVANCED INFERENCE PIPELINE")
print("="*70)

# ============================================================================
# 1. NOISE REDUCTION MODULE
# ============================================================================
class NoiseReductionModule:
    """
    Advanced noise reduction using spectral gating and denoising
    """
    def __init__(self, sr: int = 22050):
        self.sr = sr
        
    def spectral_gating(self, y: np.ndarray, threshold_db: float = -40) -> np.ndarray:
        """
        Spectral gating: suppress frequencies below threshold
        
        Args:
            y: Audio signal
            threshold_db: Threshold in dB
        
        Returns:
            Denoised audio
        """
        D = librosa.stft(y)
        magnitude = np.abs(D)
        
        # Compute threshold
        db = librosa.power_to_db(magnitude**2, ref=np.max)
        threshold = np.percentile(db, 15)  # 15th percentile
        
        # Create mask
        mask = db > threshold
        D_masked = D * mask
        
        # Reconstruct
        y_denoised = librosa.istft(D_masked)
        return y_denoised
    
    def wiener_filter(self, y: np.ndarray, noise_profile_duration: float = 0.5) -> np.ndarray:
        """
        Wiener filtering for noise reduction
        """
        # Estimate noise from first part
        noise_len = int(self.sr * noise_profile_duration)
        noise = y[:noise_len]
        
        # Compute noise power spectrum
        D_noise = librosa.stft(noise)
        noise_power = np.abs(D_noise) ** 2
        
        D = librosa.stft(y)
        magnitude = np.abs(D)
        
        # Wiener filter coefficient
        wiener_coeff = magnitude ** 2 / (magnitude ** 2 + np.mean(noise_power, axis=1, keepdims=True))
        wiener_coeff = np.maximum(wiener_coeff, 0.1)  # Floor
        
        D_filtered = D * wiener_coeff
        y_denoised = librosa.istft(D_filtered)
        
        return y_denoised
    
    def denoise(self, y: np.ndarray, method: str = "spectral_gating") -> np.ndarray:
        """
        Apply noise reduction
        """
        if method == "spectral_gating":
            return self.spectral_gating(y)
        elif method == "wiener":
            return self.wiener_filter(y)
        else:
            return y

# ============================================================================
# 2. EMOTION/PROSODY ENHANCEMENT MODULE
# ============================================================================
class EmotionEnhancementModule:
    """
    Enhance prosody and emotional expressiveness
    """
    def __init__(self, sr: int = 22050):
        self.sr = sr
    
    def enhance_prosody(self, y: np.ndarray, 
                       pitch_shift: float = 0.0,
                       duration_scale: float = 1.0,
                       energy_scale: float = 1.0) -> np.ndarray:
        """
        Enhance prosody:
        - Pitch shifting for emotional variation
        - Time stretching for speech rate variation
        - Energy scaling for emphasis
        """
        # Apply energy scaling
        y_scaled = y * energy_scale
        
        # Apply pitch shifting
        if abs(pitch_shift) > 0.01:
            y_scaled = librosa.effects.pitch_shift(
                y_scaled, sr=self.sr, n_steps=pitch_shift
            )
        
        # Apply time stretching
        if abs(duration_scale - 1.0) > 0.01:
            y_scaled = librosa.effects.time_stretch(y_scaled, rate=duration_scale)
        
        return y_scaled
    
    def add_emphasis(self, y: np.ndarray, emphasis_factor: float = 1.2) -> np.ndarray:
        """
        Add emphasis by increasing energy in key frequencies
        """
        D = librosa.stft(y)
        magnitude = np.abs(D)
        
        # Enhance frequency bands associated with emphasis
        # Typically mid-to-high frequencies
        freq_bins = librosa.fft_frequencies(sr=self.sr, n_fft=2048)
        emphasis_mask = np.where((freq_bins > 1000) & (freq_bins < 4000), emphasis_factor, 1.0)
        
        D_enhanced = D * emphasis_mask[:, np.newaxis]
        y_enhanced = librosa.istft(D_enhanced)
        
        return y_enhanced
    
    def apply_emotion(self, y: np.ndarray, emotion: str = "neutral") -> np.ndarray:
        """
        Apply different emotions:
        - neutral: no modification
        - happy: higher pitch, faster speech
        - sad: lower pitch, slower speech
        - angry: higher energy, faster speech
        - calm: lower energy, slower speech
        """
        emotion_params = {
            "happy": {"pitch": 2.0, "duration": 0.9, "energy": 1.2},
            "sad": {"pitch": -1.5, "duration": 1.2, "energy": 0.8},
            "angry": {"pitch": 1.0, "duration": 0.8, "energy": 1.4},
            "calm": {"pitch": -0.5, "duration": 1.1, "energy": 0.9},
            "neutral": {"pitch": 0.0, "duration": 1.0, "energy": 1.0}
        }
        
        params = emotion_params.get(emotion, emotion_params["neutral"])
        return self.enhance_prosody(
            y,
            pitch_shift=params["pitch"],
            duration_scale=params["duration"],
            energy_scale=params["energy"]
        )

# ============================================================================
# 3. SPEECH QUALITY ASSESSMENT MODULE
# ============================================================================
class SpeechQualityAssessment:
    """
    Compute speech quality metrics
    """
    def __init__(self, sr: int = 22050):
        self.sr = sr
    
    def compute_snr(self, y: np.ndarray) -> float:
        """Signal-to-Noise Ratio estimation"""
        # Estimate noise from silent frames
        energy = np.sqrt(np.array([
            np.sum(y[i:i+512]**2) for i in range(0, len(y)-512, 512)
        ]))
        
        noise_energy = np.percentile(energy, 15)
        signal_energy = np.mean(energy)
        
        snr = 10 * np.log10((signal_energy ** 2) / (noise_energy ** 2 + 1e-8))
        return float(snr)
    
    def compute_cepstral_distortion(self, y1: np.ndarray, y2: np.ndarray) -> float:
        """
        Mel-Cepstral Distortion (MCD) - compares reference and synthetic speech
        Lower is better (0 is identical)
        """
        # Compute MFCCs
        mfcc1 = librosa.feature.mfcc(y=y1, sr=self.sr, n_mfcc=13)
        mfcc2 = librosa.feature.mfcc(y=y2, sr=self.sr, n_mfcc=13)
        
        # Compute DTW cost
        if mfcc1.shape[1] != mfcc2.shape[1]:
            min_len = min(mfcc1.shape[1], mfcc2.shape[1])
            mfcc1 = mfcc1[:, :min_len]
            mfcc2 = mfcc2[:, :min_len]
        
        # Mean squared error
        mcd = np.sqrt(np.mean((mfcc1 - mfcc2) ** 2))
        return float(mcd)
    
    def compute_intelligibility_score(self, y: np.ndarray) -> float:
        """
        Estimate intelligibility based on formant clarity
        Range: 0-1 (higher is better)
        """
        # Compute spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Intelligibility: how clear are the prominent frequencies
        # Based on energy concentration in formant regions
        formant_region = S_db[40:80, :]  # Mid frequencies
        non_formant = np.concatenate([S_db[:40, :], S_db[80:, :]], axis=0)
        
        clarity = np.mean(formant_region) - np.mean(non_formant)
        score = 1.0 / (1.0 + np.exp(-clarity / 10.0))  # Sigmoid normalization
        
        return float(np.clip(score, 0, 1))
    
    def assess_quality(self, y: np.ndarray, reference: Optional[np.ndarray] = None) -> dict:
        """
        Comprehensive quality assessment
        """
        metrics = {
            "snr_db": self.compute_snr(y),
            "intelligibility_score": self.compute_intelligibility_score(y),
            "duration_s": librosa.get_duration(y=y, sr=self.sr),
            "mean_energy": float(np.sqrt(np.mean(y**2))),
            "peak_energy": float(np.max(np.abs(y)))
        }
        
        if reference is not None:
            metrics["mel_cepstral_distortion"] = self.compute_cepstral_distortion(reference, y)
        
        return metrics

# ============================================================================
# 4. INFERENCE ENGINE
# ============================================================================
class KannadaTTSInference:
    """
    Complete Kannada TTS inference engine
    """
    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.noise_reducer = NoiseReductionModule(sr=22050)
        self.emotion_enhancer = EmotionEnhancementModule(sr=22050)
        self.quality_assessor = SpeechQualityAssessment(sr=22050)
        
        # Try loading custom model, fallback to pretrained
        try:
            from TTS.api import TTS
            if model_path and os.path.exists(model_path):
                print(f"\n📦 Loading custom model: {model_path}")
                self.tts = TTS(
                    model_path=model_path,
                    config_path=config_path or "config/tacotron2.json",
                    gpu=(self.device == "cuda")
                )
            else:
                print("⚠️  No custom model found. Loading pretrained model...")
                self.tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=(self.device == "cuda"))
        except ImportError:
            print("❌ TTS library not installed")
            raise
    
    def synthesize(self, text: str, emotion: str = "neutral", 
                   denoise: bool = True, enhance: bool = True) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech with advanced processing
        """
        print(f"\n📝 Synthesizing: '{text}'")
        
        # Synthesize
        wav = self.tts.tts(text)
        
        if isinstance(wav, list):
            wav = np.array(wav)
        
        sr = self.tts.synthesizer.output_sample_rate
        
        # Noise reduction
        if denoise:
            print("  🎚️  Applying noise reduction...")
            wav = self.noise_reducer.denoise(wav, method="spectral_gating")
        
        # Emotion enhancement
        if enhance:
            print(f"  😊 Applying {emotion} emotion enhancement...")
            wav = self.emotion_enhancer.apply_emotion(wav, emotion=emotion)
        
        return wav, sr
    
    def assess_and_synthesize(self, text: str, output_path: str,
                             emotion: str = "neutral") -> dict:
        """
        Synthesize and assess quality
        """
        # Synthesize
        wav, sr = self.synthesize(text, emotion=emotion, denoise=True, enhance=True)
        
        # Normalize and prevent clipping
        max_val = np.max(np.abs(wav))
        if max_val > 0:
            wav = wav / max_val * 0.95  # Leave headroom
        
        # Assess quality
        quality_metrics = self.quality_assessor.assess_quality(wav)
        
        # Save
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        sf.write(output_path, wav, sr)
        
        return {
            "text": text,
            "emotion": emotion,
            "output_path": output_path,
            "sample_rate": sr,
            "quality_metrics": quality_metrics,
            "timestamp": datetime.now().isoformat()
        }

# ============================================================================
# 5. MAIN INFERENCE
# ============================================================================
def main():
    print("\n" + "="*70)
    print("INITIALIZING INFERENCE ENGINE")
    print("="*70)
    
    os.makedirs("output/inference", exist_ok=True)
    
    # Initialize engine
    engine = KannadaTTSInference(
        model_path="output/tacotron2/best_model.pth",
        config_path="config/tacotron2.json"
    )
    
    # Test sentences with different emotions
    test_cases = [
        {
            "text": "ನಮಸ್ಕಾರ, ಇದು ಕನ್ನಡ ಪಠ್ಯ-ವಾಕ್ಯ ಸಂಶ್ಲೇಷಣೆ ವ್ಯವಸ್ಥೆ.",
            "emotion": "neutral",
            "filename": "test_neutral.wav"
        },
        {
            "text": "ಬೆಂಗಳೂರು ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ!",
            "emotion": "happy",
            "filename": "test_happy.wav"
        },
        {
            "text": "ಈ ಅತ್ಯುತ್ತಮ ಪ್ರಯೋಜನ ಹೊಂದಿದ್ದೀರಿ.",
            "emotion": "calm",
            "filename": "test_calm.wav"
        }
    ]
    
    print("\n" + "="*70)
    print("GENERATING SPEECH SAMPLES")
    print("="*70)
    
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Processing: {test_case['emotion'].upper()}")
        
        output_path = f"output/inference/{test_case['filename']}"
        
        try:
            result = engine.assess_and_synthesize(
                text=test_case['text'],
                output_path=output_path,
                emotion=test_case['emotion']
            )
            results.append(result)
            
            print(f"  ✅ Saved: {output_path}")
            print(f"  📊 Quality Metrics:")
            for metric, value in result['quality_metrics'].items():
                if isinstance(value, float):
                    print(f"     {metric}: {value:.4f}")
                else:
                    print(f"     {metric}: {value}")
        
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
    
    # Save results
    import json
    with open("output/inference/results.json", "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*70)
    print("✅ INFERENCE COMPLETE!")
    print("="*70)
    print(f"\n📁 Output files saved to: output/inference/")
    print(f"📊 Results summary: output/inference/results.json")

if __name__ == "__main__":
    main()
