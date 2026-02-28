"""
Performance Evaluation Module - Kannada TTS System
Comprehensive metrics for analyzing naturalness, quality, and emotional accuracy
"""

import numpy as np
import librosa
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class SpeechQualityMetrics:
    """Computes standard speech quality metrics for TTS evaluation"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sr = sample_rate
        self.metrics_history = []
    
    def mel_cepstral_distortion(self, mel_ref: np.ndarray, mel_gen: np.ndarray) 
                                -> float:
        """
        Compute Mel-Cepstral Distortion (MCD) - measures spectral difference
        
        Lower MCD indicates better quality. Typical range: 3-6 dB
        
        Args:
            mel_ref: Reference mel-spectrogram (freq, time)
            mel_gen: Generated mel-spectrogram (freq, time)
        
        Returns:
            MCD in dB
        """
        # Ensure same length
        min_len = min(mel_ref.shape[1], mel_gen.shape[1])
        mel_ref = mel_ref[:, :min_len]
        mel_gen = mel_gen[:, :min_len]
        
        # Compute MCD as mean squared difference
        mcd = np.sqrt(2.0 * np.mean((mel_ref - mel_gen) ** 2))
        
        logger.info(f"MCD computed: {mcd:.4f} dB")
        return float(mcd)
    
    def signal_to_noise_ratio(self, audio_clean: np.ndarray, 
                             audio_noisy: np.ndarray) -> float:
        """
        Compute Signal-to-Noise Ratio (SNR)
        
        SNR = 10 * log10(signal_power / noise_power)
        Higher SNR indicates less noise. Typical range: 15-30 dB
        
        Args:
            audio_clean: Clean/reference audio
            audio_noisy: Noisy/test audio
        
        Returns:
            SNR in dB
        """
        if len(audio_clean) != len(audio_noisy):
            min_len = min(len(audio_clean), len(audio_noisy))
            audio_clean = audio_clean[:min_len]
            audio_noisy = audio_noisy[:min_len]
        
        signal = np.sum(audio_clean ** 2)
        noise = np.sum((audio_clean - audio_noisy) ** 2)
        
        if noise == 0:
            snr = 100.0  # Perfect match
        else:
            snr = 10.0 * np.log10(signal / noise)
        
        logger.info(f"SNR computed: {snr:.4f} dB")
        return float(max(snr, 0))
    
    def spectral_distortion(self, spec_ref: np.ndarray, 
                           spec_gen: np.ndarray) -> float:
        """
        Compute Spectral Distortion (SD) - frame-based spectral error
        
        Lower is better. Typical range: 2-8 dB
        
        Args:
            spec_ref: Reference spectrogram
            spec_gen: Generated spectrogram
        
        Returns:
            Mean spectral distortion in dB
        """
        min_len = min(spec_ref.shape[1], spec_gen.shape[1])
        spec_ref = spec_ref[:, :min_len]
        spec_gen = spec_gen[:, :min_len]
        
        # L2 distance for each frame
        frame_distances = np.sqrt(np.sum((spec_ref - spec_gen) ** 2, axis=0))
        mean_sd = np.mean(frame_distances)
        
        logger.info(f"Spectral Distortion computed: {mean_sd:.4f} dB")
        return float(mean_sd)
    
    def log_spectral_distance(self, spec_ref: np.ndarray, 
                             spec_gen: np.ndarray) -> float:
        """
        Compute Log Spectral Distance (LSD) - sensitive to human perception
        
        LSD = sqrt(mean((log(spec_ref) - log(spec_gen))^2))
        
        Args:
            spec_ref: Reference spectrogram (power or magnitude)
            spec_gen: Generated spectrogram
        
        Returns:
            LSD in dB
        """
        # Avoid log(0)
        eps = 1e-10
        
        spec_ref = np.clip(spec_ref, eps, None)
        spec_gen = np.clip(spec_gen, eps, None)
        
        min_len = min(spec_ref.shape[1], spec_gen.shape[1])
        spec_ref = spec_ref[:, :min_len]
        spec_gen = spec_gen[:, :min_len]
        
        lsd = np.sqrt(np.mean((np.log(spec_ref + eps) - 
                              np.log(spec_gen + eps)) ** 2))
        
        logger.info(f"Log Spectral Distance computed: {lsd:.4f} dB")
        return float(lsd)
    
    def prosody_similarity(self, f0_ref: np.ndarray, f0_gen: np.ndarray,
                          duration_ref: np.ndarray, 
                          duration_gen: np.ndarray) -> Dict[str, float]:
        """
        Compute prosody similarity - compares pitch and duration
        
        Args:
            f0_ref: Reference fundamental frequency
            f0_gen: Generated fundamental frequency
            duration_ref: Reference phone durations
            duration_gen: Generated phone durations
        
        Returns:
            Dictionary with pitch_corr and duration_rmse
        """
        # Ensure same length
        min_len = min(len(f0_ref), len(f0_gen))
        f0_ref = f0_ref[:min_len]
        f0_gen = f0_gen[:min_len]
        
        # Pitch correlation (ignore zero frames)
        mask = (f0_ref > 0) & (f0_gen > 0)
        if np.sum(mask) > 1:
            pitch_corr = float(np.corrcoef(f0_ref[mask], f0_gen[mask])[0, 1])
            pitch_corr = np.nan_to_num(pitch_corr, nan=0.0, posinf=1.0, neginf=-1.0)
        else:
            pitch_corr = 0.0
        
        # Duration RMSE
        if len(duration_ref) == len(duration_gen):
            duration_rmse = float(np.sqrt(np.mean(
                (duration_ref - duration_gen) ** 2
            )))
        else:
            duration_rmse = 999.0  # Large error if different lengths
        
        logger.info(f"Prosody - Pitch corr: {pitch_corr:.4f}, "
                   f"Duration RMSE: {duration_rmse:.4f}")
        
        return {
            'pitch_correlation': float(pitch_corr),
            'duration_rmse': float(duration_rmse)
        }
    
    def intelligibility_score(self, mel_spec: np.ndarray) -> float:
        """
        Estimate intelligibility from spectral characteristics
        
        Based on spectral stability and energy distribution
        
        Args:
            mel_spec: Mel-spectrogram
        
        Returns:
            Intelligibility score (0-1)
        """
        if mel_spec.shape[1] < 2:
            return 0.0
        
        # Spectral stability (variance over time)
        spectral_variance = np.std(mel_spec, axis=1)
        stability = 1.0 - np.mean(np.clip(spectral_variance / 
                                         (np.max(mel_spec) + 1e-10), 0, 1))
        
        # Energy distribution (how concentrated)
        energy_per_frame = np.sum(mel_spec, axis=0)
        energy_smoothness = 1.0 - np.mean(np.abs(np.diff(
            energy_per_frame / (np.max(energy_per_frame) + 1e-10)
        )))
        
        intelligibility = (stability + energy_smoothness) / 2.0
        intelligibility = np.clip(intelligibility, 0, 1)
        
        logger.info(f"Intelligibility score: {intelligibility:.4f}")
        return float(intelligibility)
    
    def naturalness_score(self, mel_spec: np.ndarray) -> float:
        """
        Estimate naturalness from spectral continuity
        
        Based on smoothness of spectral transitions
        
        Args:
            mel_spec: Mel-spectrogram
        
        Returns:
            Naturalness score (0-1)
        """
        if mel_spec.shape[1] < 2:
            return 0.0
        
        # Frame-to-frame spectral distance
        spectral_diff = np.mean(np.abs(np.diff(mel_spec, axis=1)))
        max_diff = np.max(mel_spec)
        
        # Normalize: lower variation = more natural
        normalized_diff = spectral_diff / (max_diff + 1e-10)
        naturalness = np.exp(-normalized_diff)  # Gaussian penalty
        
        naturalness = np.clip(naturalness, 0, 1)
        
        logger.info(f"Naturalness score: {naturalness:.4f}")
        return float(naturalness)


class EmotionalAccuracyMetrics:
    """Metrics for evaluating emotional expression accuracy"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sr = sample_rate
        self.emotions = ['neutral', 'happy', 'sad', 'angry', 'surprised']
    
    def prosody_diversity(self, audio: np.ndarray, 
                         emotion_style: str = 'neutral') -> Dict[str, float]:
        """
        Measures diversity in prosody - important for emotional variation
        
        Args:
            audio: Audio signal
            emotion_style: Target emotion style
        
        Returns:
            Prosody characteristics
        """
        try:
            # Extract pitch
            D = librosa.stft(audio)
            magnitude = np.abs(D)
            frequencies = librosa.fft_frequencies(sr=self.sr)
            
            # Estimate fundamental frequency region (lower frequencies)
            fundamental_region = magnitude[:100, :]
            pitch_energy = np.mean(fundamental_region, axis=0)
            
            # Pitch statistics
            pitch_var = np.std(pitch_energy)
            pitch_range = np.max(pitch_energy) - np.min(pitch_energy)
            pitch_dynamics = float(pitch_var / (np.mean(pitch_energy) + 1e-10))
            
            return {
                'pitch_variance': float(pitch_var),
                'pitch_range': float(pitch_range),
                'pitch_dynamics': pitch_dynamics,
                'emotion_style': emotion_style
            }
        except Exception as e:
            logger.warning(f"Could not compute prosody diversity: {e}")
            return {}
    
    def energy_variation(self, audio: np.ndarray) -> float:
        """
        Compute energy variation - indicates emotional expressiveness
        
        Args:
            audio: Audio signal
        
        Returns:
            Energy variation normalized (0-1)
        """
        # Frame-based energy
        frame_length = 2048
        hop_length = 512
        frames = librosa.util.frame(audio, frame_length=frame_length,
                                    hop_length=hop_length)
        energy = np.sqrt(np.mean(frames ** 2, axis=0))
        
        if np.max(energy) == 0:
            return 0.0
        
        # Normalized variation
        variation = np.std(energy) / (np.mean(energy) + 1e-10)
        variation = np.clip(variation, 0, 1)
        
        logger.info(f"Energy variation: {variation:.4f}")
        return float(variation)
    
    def emotion_consistency(self, mel_specs_by_emotion: Dict[str, np.ndarray]) 
                           -> Dict[str, float]:
        """
        Measure consistency within each emotion category
        
        Args:
            mel_specs_by_emotion: Dict of emotion -> list of mel-spectrograms
        
        Returns:
            Consistency score per emotion
        """
        consistency_scores = {}
        
        for emotion, mel_list in mel_specs_by_emotion.items():
            if len(mel_list) < 2:
                consistency_scores[emotion] = 0.0
                continue
            
            # Pairwise distances
            distances = []
            for i in range(len(mel_list) - 1):
                for j in range(i + 1, len(mel_list)):
                    mel1 = mel_list[i]
                    mel2 = mel_list[j]
                    
                    min_len = min(mel1.shape[1], mel2.shape[1])
                    mel1 = mel1[:, :min_len]
                    mel2 = mel2[:, :min_len]
                    
                    dist = np.sqrt(np.mean((mel1 - mel2) ** 2))
                    distances.append(dist)
            
            if distances:
                consistency = 1.0 / (1.0 + np.mean(distances))
                consistency_scores[emotion] = float(
                    np.clip(consistency, 0, 1)
                )
            else:
                consistency_scores[emotion] = 0.0
        
        logger.info(f"Emotion consistency: {consistency_scores}")
        return consistency_scores


class ComparativeAnalysis:
    """Compare VITS (hybrid) vs Tacotron2 (non-hybrid) performance"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sr = sample_rate
        self.results = {}
    
    def compare_approaches(self, 
                          audio_vits: np.ndarray,
                          audio_tacotron2: np.ndarray,
                          reference_audio: Optional[np.ndarray] = None
                          ) -> Dict:
        """
        Comprehensive comparison between VITS and Tacotron2
        
        Args:
            audio_vits: Generated audio using VITS (hybrid)
            audio_tacotron2: Generated audio using Tacotron2 (non-hybrid)
            reference_audio: Optional reference for quality comparison
        
        Returns:
            Comparison results
        """
        quality_metrics = SpeechQualityMetrics(self.sr)
        
        # Extract mel-spectrograms
        try:
            mel_vits = librosa.feature.melspectrogram(y=audio_vits, sr=self.sr)
            mel_tacotron2 = librosa.feature.melspectrogram(
                y=audio_tacotron2, sr=self.sr
            )
            
            comparison = {
                'timestamp': datetime.now().isoformat(),
                'vits': {
                    'audio_length': len(audio_vits) / self.sr,
                    'intelligibility': quality_metrics.intelligibility_score(
                        mel_vits
                    ),
                    'naturalness': quality_metrics.naturalness_score(mel_vits),
                    'energy_variation': self._compute_energy_variation(
                        audio_vits
                    ),
                },
                'tacotron2': {
                    'audio_length': len(audio_tacotron2) / self.sr,
                    'intelligibility': quality_metrics.intelligibility_score(
                        mel_tacotron2
                    ),
                    'naturalness': quality_metrics.naturalness_score(
                        mel_tacotron2
                    ),
                    'energy_variation': self._compute_energy_variation(
                        audio_tacotron2
                    ),
                }
            }
            
            # Direct comparison
            comparison['comparative'] = {
                'spectral_distortion': quality_metrics.spectral_distortion(
                    mel_vits, mel_tacotron2
                ),
                'intelligibility_advantage': (
                    comparison['vits']['intelligibility'] - 
                    comparison['tacotron2']['intelligibility']
                ),
                'naturalness_advantage': (
                    comparison['vits']['naturalness'] - 
                    comparison['tacotron2']['naturalness']
                ),
            }
            
            # If reference available
            if reference_audio is not None:
                mel_ref = librosa.feature.melspectrogram(
                    y=reference_audio, sr=self.sr
                )
                comparison['vs_reference'] = {
                    'vits_mcd': quality_metrics.mel_cepstral_distortion(
                        mel_ref, mel_vits
                    ),
                    'vits_lsd': quality_metrics.log_spectral_distance(
                        mel_ref, mel_vits
                    ),
                    'tacotron2_mcd': quality_metrics.mel_cepstral_distortion(
                        mel_ref, mel_tacotron2
                    ),
                    'tacotron2_lsd': quality_metrics.log_spectral_distance(
                        mel_ref, mel_tacotron2
                    ),
                }
                
                # Quality improvement
                mcd_improvement = (
                    comparison['vs_reference']['tacotron2_mcd'] - 
                    comparison['vs_reference']['vits_mcd']
                ) / comparison['vs_reference']['tacotron2_mcd'] * 100
                
                comparison['quality_improvement_percent'] = float(
                    np.clip(mcd_improvement, 0, 100)
                )
            
            logger.info("Comparative analysis complete")
            self.results = comparison
            return comparison
            
        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            return {}
    
    def _compute_energy_variation(self, audio: np.ndarray) -> float:
        """Helper: compute energy variation"""
        frame_length = 2048
        hop_length = 512
        frames = librosa.util.frame(audio, frame_length=frame_length,
                                    hop_length=hop_length)
        energy = np.sqrt(np.mean(frames ** 2, axis=0))
        
        if np.max(energy) == 0:
            return 0.0
        
        return float(np.std(energy) / (np.mean(energy) + 1e-10))
    
    def export_results(self, output_path: str):
        """Export comparison results to JSON"""
        try:
            with open(output_path, 'w') as f:
                # Convert numpy types for JSON serialization
                results = self._serialize_results(self.results)
                json.dump(results, f, indent=2)
            logger.info(f"Results exported to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
    
    def _serialize_results(self, obj):
        """Convert numpy types for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._serialize_results(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_results(item) for item in obj]
        return obj


class EvaluationPipeline:
    """Complete evaluation pipeline for Kannada TTS"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sr = sample_rate
        self.quality_metrics = SpeechQualityMetrics(sample_rate)
        self.emotion_metrics = EmotionalAccuracyMetrics(sample_rate)
        self.comparison = ComparativeAnalysis(sample_rate)
    
    def evaluate_synthesis(self, 
                          audio: np.ndarray,
                          emotion: str = 'neutral',
                          reference_audio: Optional[np.ndarray] = None
                          ) -> Dict:
        """
        Full evaluation of synthesized audio
        
        Args:
            audio: Synthesized audio
            emotion: Emotion category
            reference_audio: Optional reference audio
        
        Returns:
            Complete evaluation report
        """
        try:
            mel = librosa.feature.melspectrogram(y=audio, sr=self.sr)
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'audio_length_seconds': len(audio) / self.sr,
                'emotion': emotion,
                'quality_metrics': {
                    'intelligibility': self.quality_metrics.intelligibility_score(
                        mel
                    ),
                    'naturalness': self.quality_metrics.naturalness_score(mel),
                },
                'emotional_metrics': {
                    'energy_variation': self.emotion_metrics.energy_variation(audio),
                    'prosody': self.emotion_metrics.prosody_diversity(
                        audio, emotion
                    ),
                }
            }
            
            if reference_audio is not None:
                mel_ref = librosa.feature.melspectrogram(
                    y=reference_audio, sr=self.sr
                )
                report['quality_metrics']['mcd'] = (
                    self.quality_metrics.mel_cepstral_distortion(mel_ref, mel)
                )
                report['quality_metrics']['lsd'] = (
                    self.quality_metrics.log_spectral_distance(mel_ref, mel)
                )
                report['quality_metrics']['snr'] = (
                    self.quality_metrics.signal_to_noise_ratio(
                        reference_audio, audio
                    )
                )
            
            logger.info("Synthesis evaluation complete")
            return report
        
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {}
    
    def generate_report(self, evaluations: List[Dict], 
                       output_path: str = 'evaluation_report.json'):
        """Generate comprehensive evaluation report"""
        try:
            # Aggregate statistics
            if evaluations:
                natural_scores = [e['quality_metrics']['naturalness'] 
                                 for e in evaluations 
                                 if 'naturalness' in e.get('quality_metrics', {})]
                intel_scores = [e['quality_metrics']['intelligibility'] 
                               for e in evaluations 
                               if 'intelligibility' in e.get('quality_metrics', {})]
                
                report = {
                    'timestamp': datetime.now().isoformat(),
                    'total_evaluations': len(evaluations),
                    'aggregate_statistics': {
                        'avg_naturalness': float(np.mean(natural_scores)) 
                                          if natural_scores else 0.0,
                        'avg_intelligibility': float(np.mean(intel_scores)) 
                                              if intel_scores else 0.0,
                        'naturalness_range': (
                            float(np.min(natural_scores)),
                            float(np.max(natural_scores))
                        ) if natural_scores else (0.0, 0.0),
                    },
                    'detailed_evaluations': evaluations
                }
                
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                
                logger.info(f"Report generated: {output_path}")
                return report
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {}


# Example usage functions
def example_evaluation():
    """Example: Evaluate synthesized audio"""
    import warnings
    warnings.filterwarnings('ignore')
    
    # Initialize pipeline
    pipeline = EvaluationPipeline(sample_rate=22050)
    
    # Create synthetic test audio
    duration = 2.0
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration))
    
    # Simulate synthesized audio (sine wave with noise)
    freq = 200
    audio = np.sin(2 * np.pi * freq * t)
    audio += 0.01 * np.random.randn(len(t))
    
    # Evaluate
    evaluation = pipeline.evaluate_synthesis(
        audio, 
        emotion='neutral'
    )
    
    print("Evaluation Results:")
    print(f"  Naturalness: {evaluation['quality_metrics']['naturalness']:.4f}")
    print(f"  Intelligibility: {evaluation['quality_metrics']['intelligibility']:.4f}")
    print(f"  Energy Variation: {evaluation['emotional_metrics']['energy_variation']:.4f}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    example_evaluation()
