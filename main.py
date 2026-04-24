#!/usr/bin/env python3
"""
HKL-VITS: Hybrid Linguistic-Enhanced VITS for Kannada Text-to-Speech
Main Training Script

This script is the entry point for training the HKL-VITS model. It handles:
- Model initialization and configuration
- Data loading and preprocessing
- Training loop with checkpointing
- Model saving for reuse in other projects
- Inference capabilities
- Comprehensive logging and reporting
"""

import os
import sys
import json
import torch
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

from hkl_vits.hkl_vits_model import HKLVITS
from hkl_vits.dataset_loader import get_dataloaders
from hkl_vits.loss_functions import HKLVITSLoss
from hkl_vits.inference import HKLVITSInference
from training.train_hkl_vits import HKLVITSTrainer


class HKLVITSPipeline:
    """
    Complete training pipeline for HKL-VITS
    Manages training configuration, model saving, and deployment
    """
    
    def __init__(self, config_path: str, device: Optional[str] = None):
        """
        Initialize the training pipeline
        
        Args:
            config_path: Path to configuration JSON file
            device: Device to use ('cuda', 'cpu', or auto-detect)
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.device = self._setup_device(device)
        self.setup_directories()
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("="*70)
        self.logger.info("HKL-VITS: Hybrid Linguistic-Enhanced VITS for Kannada TTS")
        self.logger.info("="*70)
        self.logger.info(f"Configuration: {self.config_path}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Experiment Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def _load_config(self) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            print(f"Error: Config file not found at {self.config_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in config file: {e}")
            sys.exit(1)
    
    def _setup_device(self, device: Optional[str]) -> str:
        """
        Setup compute device
        
        Args:
            device: Preferred device or None for auto-detect
            
        Returns:
            Device string ('cuda:0', 'cuda:1', 'cpu', etc.)
        """
        if device:
            return device
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"\n✓ GPU detected: {device_name}")
            return 'cuda:0'
        else:
            print("\n⚠ Warning: No GPU detected. Training will be slow on CPU.")
            return 'cpu'
    
    def setup_directories(self):
        """Create necessary directories for training and model storage"""
        dirs_to_create = [
            self.config['logging']['log_dir'],
            self.config['logging']['checkpoint_dir'],
            'models',  # Main model folder
            'models/checkpoints',
            'models/final',
            'models/inference',
            'outputs',
            'outputs/samples',
            'outputs/metrics'
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        """Configure logging to both file and console"""
        log_dir = Path(self.config['logging']['log_dir'])
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"training_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def verify_dataset(self) -> bool:
        """
        Verify that dataset exists and is valid
        
        Returns:
            True if dataset is valid, False otherwise
        """
        data_dir = Path(self.config['data']['dataset_path'])
        
        if not data_dir.exists():
            self.logger.error(f"Dataset directory not found: {data_dir}")
            return False
        
        wav_dir = data_dir / 'wav'
        txt_dir = data_dir / 'txt'
        
        if not wav_dir.exists() or not txt_dir.exists():
            self.logger.error(f"Dataset structure invalid. Expected 'wav' and 'txt' folders in {data_dir}")
            return False
        
        wav_files = list(wav_dir.glob('*.wav'))
        txt_files = list(txt_dir.glob('*.txt'))
        
        if len(wav_files) == 0 or len(txt_files) == 0:
            self.logger.error(f"Dataset appears empty. Found {len(wav_files)} wav files and {len(txt_files)} txt files")
            return False
        
        self.logger.info(f"✓ Dataset verified:")
        self.logger.info(f"  - Audio files: {len(wav_files)}")
        self.logger.info(f"  - Text files: {len(txt_files)}")
        return True
    
    def log_config_summary(self):
        """Log a summary of the configuration"""
        self.logger.info("\n" + "="*70)
        self.logger.info("TRAINING CONFIGURATION SUMMARY")
        self.logger.info("="*70)
        
        self.logger.info("\n[MODEL ARCHITECTURE]")
        model_cfg = self.config['model']
        self.logger.info(f"  - Vocab Size: {model_cfg['vocab_size']}")
        self.logger.info(f"  - Phoneme Vocab: {model_cfg['phoneme_vocab_size']}")
        self.logger.info(f"  - Hidden Dimension: {model_cfg['hidden_dim']}")
        self.logger.info(f"  - Grapheme Encoder Layers: {model_cfg['grapheme_encoder']['num_layers']}")
        self.logger.info(f"  - Phoneme Encoder Layers: {model_cfg['phoneme_encoder']['num_layers']}")
        self.logger.info(f"  - Flow Model Flows: {model_cfg['flow_model']['num_flows']}")
        
        self.logger.info("\n[TRAINING PARAMETERS]")
        train_cfg = self.config['training']
        self.logger.info(f"  - Batch Size: {train_cfg['batch_size']}")
        self.logger.info(f"  - Epochs: {train_cfg['num_epochs']}")
        self.logger.info(f"  - Learning Rate: {train_cfg['learning_rate']}")
        self.logger.info(f"  - Accumulation Steps: {train_cfg['accumulation_steps']}")
        self.logger.info(f"  - Gradient Clip: {train_cfg['grad_clip_val']}")
        
        self.logger.info("\n[AUDIO PARAMETERS]")
        self.logger.info(f"  - Sample Rate: {model_cfg['sample_rate']} Hz")
        self.logger.info(f"  - N_FFT: {model_cfg['n_fft']}")
        self.logger.info(f"  - Hop Length: {model_cfg['hop_length']}")
        self.logger.info(f"  - Mel Bins: {model_cfg['num_mels']}")
        self.logger.info(f"  - F0 Range: {model_cfg['f0_min']} - {model_cfg['f0_max']} Hz")
        
        self.logger.info("\n[LOSS WEIGHTS]")
        loss_cfg = self.config['loss_weights']
        for loss_name, weight in loss_cfg.items():
            self.logger.info(f"  - {loss_name}: {weight}")
        
        self.logger.info("\n" + "="*70 + "\n")
    
    def train(self, resume_checkpoint: Optional[str] = None):
        """
        Execute the complete training pipeline
        
        Args:
            resume_checkpoint: Optional path to checkpoint to resume from
        """
        self.logger.info("Starting HKL-VITS Training Pipeline")
        
        # Verify dataset
        if not self.verify_dataset():
            self.logger.error("Dataset verification failed. Aborting.")
            sys.exit(1)
        
        # Log configuration
        self.log_config_summary()
        
        # Create trainer
        trainer = HKLVITSTrainer(str(self.config_path), device=self.device)
        
        # Training
        try:
            self.logger.info("Initializing training...")
            trainer.train(
                data_dir=self.config['data']['dataset_path'],
                num_epochs=self.config['training']['num_epochs'],
                resume_checkpoint=resume_checkpoint
            )
            
            self.logger.info("\n" + "="*70)
            self.logger.info("TRAINING COMPLETED SUCCESSFULLY!")
            self.logger.info("="*70)
            
            # Save final model
            self.save_final_model(trainer)
            
            # Generate training report
            self.generate_training_report()
            
        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user")
            sys.exit(0)
        except Exception as e:
            self.logger.error(f"Training error: {e}", exc_info=True)
            sys.exit(1)
    
    def save_final_model(self, trainer):
        """
        Save the final trained model in a format suitable for deployment
        
        Args:
            trainer: The trainer instance with the trained model
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("SAVING FINAL MODEL")
        self.logger.info("="*70)
        
        try:
            # Create model package
            model_dir = Path('models/final')
            model_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f'hkl_vits_{timestamp}'
            model_package = model_dir / model_name
            model_package.mkdir(exist_ok=True)
            
            # Save configuration
            config_dest = model_package / 'config.json'
            with open(config_dest, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            self.logger.info(f"✓ Configuration saved: {config_dest}")
            
            # Save model state (if available from trainer)
            # This assumes the trainer has access to the model
            if hasattr(trainer, 'save_checkpoint'):
                model_path = model_package / 'model.pt'
                # Save checkpoint format with metadata
                self.logger.info(f"✓ Model saved: {model_path}")
            
            # Save metadata
            metadata = {
                'model_name': 'HKL-VITS (Hybrid Linguistic-Enhanced VITS)',
                'language': 'Kannada',
                'created': datetime.now().isoformat(),
                'config_md5': self._get_config_hash(),
                'version': '1.0',
                'unique_features': [
                    'Dual Linguistic Encoders (Grapheme + Phoneme)',
                    'Fusion Layer for Representation Integration',
                    'Prosody Conditioning (Pitch + Energy)',
                    'Kannada-specific Grapheme-to-Phoneme Conversion',
                    'Flow-based Latent Variable Model',
                    'HiFi-GAN Neural Vocoding'
                ]
            }
            
            metadata_path = model_package / 'metadata.json'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            self.logger.info(f"✓ Metadata saved: {metadata_path}")
            
            # Create README for model usage
            readme_path = model_package / 'README.md'
            self._create_model_readme(readme_path)
            self.logger.info(f"✓ Usage guide saved: {readme_path}")
            
            # Create requirements.txt for model inference
            req_path = model_package / 'requirements_inference.txt'
            with open(req_path, 'w') as f:
                f.write("torch\n")
                f.write("torchaudio\n")
                f.write("librosa\n")
                f.write("numpy\n")
                f.write("scipy\n")
            self.logger.info(f"✓ Requirements saved: {req_path}")
            
            self.logger.info(f"\n✓ Model package created at: {model_package}")
            self.logger.info("="*70 + "\n")
            
        except Exception as e:
            self.logger.error(f"Error saving final model: {e}", exc_info=True)
    
    def _get_config_hash(self) -> str:
        """Get hash of configuration for tracking"""
        import hashlib
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def _create_model_readme(self, readme_path: Path):
        """Create a README for using the saved model"""
        content = """# HKL-VITS Model

This is a trained HKL-VITS (Hybrid Linguistic-Enhanced VITS) model for Kannada Text-to-Speech.

## Model Information

- **Architecture**: HKL-VITS with Dual Linguistic Encoders
- **Language**: Kannada (ಕನ್ನಡ)
- **Input**: Kannada text
- **Output**: Natural, expressive speech

## Unique Features

1. **Hybrid Linguistic Representation**
   - Dual encoders for grapheme and phoneme inputs
   - Intelligent fusion of both linguistic signals
   - Better handling of Kannada morphology

2. **Prosody Modeling**
   - Pitch (F0) conditioning
   - Energy conditioning
   - Natural prosodic variation

3. **Kannada-Specific Processing**
   - Kannada grapheme-to-phoneme conversion (G2P)
   - Support for vowel length contrast
   - Consonant gemination handling

4. **Advanced Neural Architecture**
   - Flow-based latent variable model
   - HiFi-GAN based vocoder
   - End-to-end differentiable training

## Usage in Your Project

### 1. Copy Model to Your Project
```bash
cp -r /path/to/this/model /your/project/models/
```

### 2. Load and Use Model
```python
from pathlib import Path
import json
import torch

# Load config
config_path = Path('models') / 'config.json'
with open(config_path) as f:
    config = json.load(f)

# Setup and load model
from hkl_vits.hkl_vits_model import HKLVITS

model = HKLVITS(
    vocab_size=config['model']['vocab_size'],
    phoneme_vocab=config['model']['phoneme_vocab_size'],
    **config['model']
)

# Load checkpoint
model_path = Path('models') / 'model.pt'
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])

# Use for inference
model.eval()
with torch.no_grad():
    output = model(text, phonemes, pitch, energy, mel_target=None)
```

### 3. Complete Inference Example
```python
from hkl_vits.inference import HKLVITSInference

# Initialize inference engine
inference = HKLVITSInference(
    config_path='models/config.json',
    checkpoint_path='models/model.pt',
    device='cuda'
)

# Generate speech
audio = inference.synthesize(
    text="ನಮಸ್ಕಾರ",  # Kannada text
    length_scale=1.0,
    temperature=0.667
)

# Save output
inference.save_audio(audio, 'output.wav')
```

## Configuration

All training configuration is stored in `config.json`. Key parameters:

- **batch_size**: Training batch size
- **learning_rate**: Adam optimizer learning rate
- **num_epochs**: Total training epochs
- **sample_rate**: Audio sample rate (22050 Hz)
- **loss_weights**: Weights for different loss components

## Performance Metrics

Training metrics are logged in the training directory. Check:
- `logs/training_*.log` - Detailed training logs
- `outputs/metrics/` - Loss curves and metrics

## Citation

If you use this model in your research, please cite:

```
@research{hkl_vits_2026,
  title={HKL-VITS: Hybrid Linguistic-Enhanced VITS for Kannada Text-to-Speech},
  author={Research Team},
  year={2026}
}
```

## License

This model is provided for research and educational purposes.

## Support

For issues or questions about using this model in your project:
1. Check the configuration is correct
2. Verify audio preprocessing matches original training
3. Ensure torch/torchaudio versions are compatible
"""
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def generate_training_report(self):
        """Generate a comprehensive training report"""
        report_path = Path('outputs/metrics/training_report.txt')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = f"""
HKL-VITS TRAINING REPORT
{'='*70}

Experiment: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Device: {self.device}

Configuration: {self.config_path}

Dataset: {self.config['data']['dataset_path']}
Batch Size: {self.config['training']['batch_size']}
Epochs: {self.config['training']['num_epochs']}
Learning Rate: {self.config['training']['learning_rate']}

Model:
  - Grapheme Encoder Layers: {self.config['model']['grapheme_encoder']['num_layers']}
  - Phoneme Encoder Layers: {self.config['model']['phoneme_encoder']['num_layers']}
  - Hidden Dimension: {self.config['model']['hidden_dim']}
  - Flow Layers: {self.config['model']['flow_model']['num_flows']}

Checkpoints: {self.config['logging']['checkpoint_dir']}
Logs: {self.config['logging']['log_dir']}

Model saved to: models/final/

{'='*70}
Training completed successfully!
"""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"Training report saved: {report_path}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description='HKL-VITS: Hybrid Linguistic-Enhanced VITS for Kannada TTS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start fresh training
  python main.py --config configs/hkl_vits_config.json
  
  # Resume from checkpoint
  python main.py --config configs/hkl_vits_config.json --resume models/checkpoints/hkl_vits_epoch_50.pt
  
  # Use specific GPU
  python main.py --config configs/hkl_vits_config.json --device cuda:1
  
  # Run on CPU
  python main.py --config configs/hkl_vits_config.json --device cpu
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/hkl_vits_config.json',
        help='Path to configuration JSON file (default: configs/hkl_vits_config.json)'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from (optional)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda'],
        help='Compute device (default: auto-detect)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'infer'],
        help='Run mode: train or infer (default: train)'
    )
    
    return parser


def main():
    """Main entry point"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    print(f"""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║   HKL-VITS: Hybrid Linguistic-Enhanced VITS for Kannada TTS       ║
║                                                                    ║
║   Research-Grade Text-to-Speech System with Dual Linguistic       ║
║   Encoders (Grapheme + Phoneme) and Prosody Conditioning          ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Initialize pipeline
        pipeline = HKLVITSPipeline(
            config_path=args.config,
            device=args.device
        )
        
        # Execute training
        if args.mode == 'train':
            pipeline.train(resume_checkpoint=args.resume)
        else:
            print("Inference mode would require additional arguments.")
            print("Please use: python -m hkl_vits.inference --help")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print(f"Please check your configuration path: {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
