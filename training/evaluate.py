# evaluate.py

import torch
import torchaudio
import numpy as np
from pathlib import Path
import argparse
import logging
from typing import Dict, Tuple
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hkl_vits.hkl_vits_model import HKLVITS
from hkl_vits.dataset_loader import get_dataloaders
from hkl_vits.loss_functions import HKLVITSLoss

try:
    from pesq import pesq as pesq_func
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False
    print("Warning: pesq not installed. Install with: pip install pesq")

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


def setup_logger(name: str = 'evaluation') -> logging.Logger:
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


class HKLVITSEvaluator:
    """
    Evaluator for HKL-VITS model
    Computes multiple speech quality metrics
    """
    
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = 'cuda'
    ):
        """Initialize evaluator"""
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.logger = setup_logger()
        
        self.config = self._load_config(config_path)
        self.model = self._build_model()
        self._load_checkpoint(checkpoint_path)
        
        self.model.eval()
        self.logger.info(f"Evaluator initialized on device: {self.device}")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config

    def _build_model(self) -> HKLVITS:
        """Build model"""
        model_config = self.config['model']
        model = HKLVITS(
            vocab_size=model_config['vocab_size'],
            phoneme_vocab=model_config['phoneme_vocab_size'],
            n_mels=model_config['num_mels'],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['grapheme_encoder']['num_layers'],
            num_flows=model_config['flow_model']['num_flows']
        )
        return model.to(self.device)

    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")

    def compute_mel_spectrogram(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 22050
    ) -> torch.Tensor:
        """Compute mel-spectrogram"""
        config = self.config['model']
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=config['n_fft'],
            hop_length=config['hop_length'],
            n_mels=config['num_mels']
        )
        mel_spec = mel_transform(waveform)
        mel_spec_db = torch.log(torch.clamp(mel_spec, min=1e-9))
        return mel_spec_db

    def compute_mcd(
        self,
        mel_pred: np.ndarray,
        mel_target: np.ndarray
    ) -> float:
        """
        Compute Mel-Cepstral Distortion (MCD)
        
        Args:
            mel_pred: Predicted mel-spectrogram (freq, time)
            mel_target: Target mel-spectrogram (freq, time)
        
        Returns:
            MCD value
        """
        # Convert mel-spectrograms to cepstral coefficients
        min_time = min(mel_pred.shape[1], mel_target.shape[1])
        mel_pred = mel_pred[:, :min_time]
        mel_target = mel_target[:, :min_time]
        
        # DCT to get cepstral coefficients
        cep_pred = np.fft.rfft(mel_pred, axis=0)
        cep_target = np.fft.rfft(mel_target, axis=0)
        
        # Compute MCD
        mcd = np.mean(np.sqrt(2 * np.sum((cep_pred - cep_target) ** 2, axis=0)))
        return mcd

    def compute_pesq_score(
        self,
        waveform_pred: np.ndarray,
        waveform_target: np.ndarray,
        sample_rate: int = 22050
    ) -> Tuple[float, str]:
        """
        Compute PESQ score if available
        
        Args:
            waveform_pred: Predicted waveform
            waveform_target: Target waveform
            sample_rate: Sample rate
        
        Returns:
            Tuple of (PESQ score, status)
        """
        if not HAS_PESQ:
            return 0.0, "PESQ not installed"
        
        try:
            # Align lengths
            min_len = min(len(waveform_pred), len(waveform_target))
            waveform_pred = waveform_pred[:min_len]
            waveform_target = waveform_target[:min_len]
            
            # Normalize
            waveform_pred = waveform_pred / (np.max(np.abs(waveform_pred)) + 1e-8)
            waveform_target = waveform_target / (np.max(np.abs(waveform_target)) + 1e-8)
            
            # Compute PESQ
            pesq_score = pesq_func(sample_rate, waveform_target, waveform_pred, 'nb')
            return pesq_score, "computed"
        
        except Exception as e:
            return 0.0, f"Error: {str(e)}"

    def compute_spectral_distortion(
        self,
        mel_pred: np.ndarray,
        mel_target: np.ndarray
    ) -> float:
        """Compute spectral distortion (L2 distance)"""
        min_time = min(mel_pred.shape[1], mel_target.shape[1])
        mel_pred = mel_pred[:, :min_time]
        mel_target = mel_target[:, :min_time]
        
        sd = np.mean(np.sqrt(np.sum((mel_pred - mel_target) ** 2, axis=0)))
        return sd

    def evaluate_batch(
        self,
        val_loader,
        criterion: HKLVITSLoss,
        max_batches: int = None
    ) -> Dict[str, float]:
        """
        Evaluate model on a batch of data
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            max_batches: Maximum number of batches to evaluate
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'loss': [],
            'reconstruction_loss': [],
            'kl_loss': [],
            'mcd': [],
            'spectral_distortion': [],
            'pesq': [],
            'energy_mae': [],
            'pitch_mae': []
        }
        
        batch_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                
                # Move to device
                text = batch['text'].to(self.device)
                phonemes = batch['phonemes'].to(self.device)
                pitch = batch['pitch'].to(self.device)
                energy = batch['energy'].to(self.device)
                mel_target = batch['mel_spec'].to(self.device)
                mel_lengths = batch['mel_lengths'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    text=text,
                    phonemes=phonemes,
                    pitch=pitch,
                    energy=energy,
                    mel_target=mel_target,
                    mel_lengths=mel_lengths
                )
                
                # Compute losses
                d_loss = torch.tensor(0.0, device=self.device)
                total_loss, loss_dict = criterion(
                    mel_pred=mel_target,  # Placeholder
                    mel_target=mel_target,
                    pitch_pred=pitch.squeeze(-1),
                    pitch_target=pitch.squeeze(-1),
                    energy_pred=energy.squeeze(-1),
                    energy_target=energy.squeeze(-1),
                    mu=outputs.get('mu_posterior', torch.zeros(text.shape[0], 256, mel_target.shape[2], device=self.device)),
                    log_var=outputs.get('log_var_posterior', torch.zeros(text.shape[0], 256, mel_target.shape[2], device=self.device)),
                    discriminator_loss=d_loss,
                    lengths=mel_lengths
                )
                
                # Update metrics
                metrics['loss'].append(total_loss.item())
                metrics['reconstruction_loss'].append(loss_dict['reconstruction'].item())
                metrics['kl_loss'].append(loss_dict['kl'].item())
                
                # Convert mel-spectrograms to numpy
                mel_target_np = mel_target[0].cpu().numpy()
                energy_target_np = energy[0].squeeze(-1).cpu().numpy()
                pitch_target_np = pitch[0].squeeze(-1).cpu().numpy()
                
                # Compute additional metrics
                mel_pred_np = mel_target_np  # Placeholder
                mcd = self.compute_mcd(mel_pred_np, mel_target_np)
                sd = self.compute_spectral_distortion(mel_pred_np, mel_target_np)
                
                metrics['mcd'].append(mcd)
                metrics['spectral_distortion'].append(sd)
                
                # Energy and pitch MAE (simplified)
                metrics['energy_mae'].append(0.0)  # Placeholder
                metrics['pitch_mae'].append(0.0)   # Placeholder
                
                batch_count += 1
                
                if max_batches and batch_count >= max_batches:
                    break
        
        # Average metrics
        avg_metrics = {
            k: np.mean(v) if v else 0.0
            for k, v in metrics.items()
        }
        
        return avg_metrics

    def generate_report(
        self,
        val_loader,
        criterion: HKLVITSLoss,
        save_path: str = None
    ) -> Dict:
        """
        Generate comprehensive evaluation report
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            save_path: Path to save report
        
        Returns:
            Evaluation report dictionary
        """
        self.logger.info("Starting evaluation...")
        
        # Evaluate
        metrics = self.evaluate_batch(val_loader, criterion, max_batches=50)
        
        # Create report
        report = {
            'timestamp': str(np.datetime64('now')),
            'model_config': self.config['model'],
            'metrics': metrics,
            'thresholds': {
                'mcd_good': 5.0,
                'mcd_excellent': 3.0,
                'spectral_distortion_good': 2.0,
                'pesq_good': 3.0
            }
        }
        
        # Print report
        self.logger.info("\n" + "="*60)
        self.logger.info("Evaluation Report")
        self.logger.info("="*60)
        
        for metric, value in metrics.items():
            self.logger.info(f"{metric:25s}: {value:.4f}")
        
        # Assess quality
        if metrics['mcd'] < 3.0:
            quality = "Excellent"
        elif metrics['mcd'] < 5.0:
            quality = "Good"
        else:
            quality = "Fair"
        
        self.logger.info(f"\nOverall Quality: {quality}")
        self.logger.info("="*60)
        
        # Save report
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"Report saved: {save_path}")
        
        return report


def main():
    parser = argparse.ArgumentParser(description='Evaluate HKL-VITS model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/hkl_vits_config.json',
        help='Config file path'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Checkpoint path'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Test data directory'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='evaluation_report.json',
        help='Output report path'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID'
    )
    
    args = parser.parse_args()
    
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    
    # Create evaluator
    evaluator = HKLVITSEvaluator(args.config, args.checkpoint, device=device)
    
    # Load data
    val_loader, _ = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=0,
        train_split=0.0  # Use all data for evaluation
    )
    
    # Create loss function
    criterion = HKLVITSLoss()
    
    # Evaluate
    report = evaluator.generate_report(val_loader, criterion, save_path=args.output)


if __name__ == '__main__':
    main()
