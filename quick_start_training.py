#!/usr/bin/env python3
"""
Quick Start Guide for HKL-VITS Training

This script demonstrates how to use the main.py training pipeline
to train and save the Kannada HKL-VITS model for reuse in other projects.
"""

import os
import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def verify_environment():
    """Verify Python environment and dependencies"""
    print_header("STEP 1: Verifying Environment")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("[FAILED] Python 3.8+ required")
        return False
    print(f"[OK] Python {sys.version.split()[0]}")
    
    # Check key packages
    packages = ['torch', 'torchaudio', 'librosa', 'numpy']
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"[OK] {pkg} installed")
        except ImportError:
            print(f"[FAILED] {pkg} not found - run: pip install {pkg}")
            return False
    
    return True


def verify_data():
    """Verify dataset exists"""
    print_header("STEP 2: Verifying Dataset")
    
    data_path = Path('data/kannada_tts_dataset')
    
    if not data_path.exists():
        print(f"[WARNING] Dataset not found at {data_path}")
        print("\nTo download the dataset:")
        print(f"  python dataset.py full")
        return False
    
    wav_dir = data_path / 'wav'
    txt_dir = data_path / 'txt'
    
    if not (wav_dir.exists() and txt_dir.exists()):
        print("[FAILED] Invalid dataset structure")
        return False
    
    wav_files = list(wav_dir.glob('*.wav'))
    txt_files = list(txt_dir.glob('*.txt'))
    
    print(f"[OK] Dataset found at: {data_path}")
    print(f"  - Audio files: {len(wav_files)}")
    print(f"  - Text files: {len(txt_files)}")
    
    if len(wav_files) < 100:
        print("[WARNING] Dataset has fewer than 100 samples (training may be suboptimal)")
    
    return True


def check_config():
    """Check configuration file"""
    print_header("STEP 3: Verifying Configuration")
    
    config_path = Path('configs/hkl_vits_config.json')
    
    if not config_path.exists():
        print(f"[FAILED] Config file not found: {config_path}")
        return False
    
    print(f"[OK] Configuration file found: {config_path}")
    print("\nKey settings:")
    
    import json
    with open(config_path) as f:
        config = json.load(f)
    
    print(f"  - Model hidden dim: {config['model']['hidden_dim']}")
    print(f"  - Batch size: {config['training']['batch_size']}")
    print(f"  - Learning rate: {config['training']['learning_rate']}")
    print(f"  - Epochs: {config['training']['num_epochs']}")
    print(f"  - Sample rate: {config['model']['sample_rate']} Hz")
    
    return True


def check_gpu():
    """Check GPU availability"""
    print_header("STEP 4: Checking GPU Availability")
    
    import torch
    
    if torch.cuda.is_available():
        print(f"[OK] GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA Version: {torch.version.cuda}")
        print(f"  - cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"  - Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("[WARNING] GPU not available - Training will use CPU (significantly slower)")
        print("  For GPU training, install CUDA and PyTorch with CUDA support")
        return False


def start_training():
    """Start the training process"""
    print_header("STEP 5: Starting Training")
    
    print("Starting HKL-VITS training pipeline...")
    print("\nCommand: python main.py")
    print("\nTraining will:")
    print("  1. Load configuration from configs/hkl_vits_config.json")
    print("  2. Load dataset from data/kannada_tts_dataset/")
    print("  3. Initialize model with dual encoders")
    print("  4. Train for specified epochs")
    print("  5. Save checkpoints every N epochs")
    print("  6. Save final model to models/final/")
    
    print("\n" + "="*70)
    print("Monitoring Training:")
    print("="*70)
    print("\nReal-time Logs:")
    print("  logs/training_YYYYMMDD_HHMMSS.log")
    print("\nTensorBoard (if available):")
    print("  tensorboard --logdir logs/")
    print("\nCheckpoints saved to:")
    print("  models/checkpoints/hkl_vits_epoch_*.pt")
    print("\nFinal model saved to:")
    print("  models/final/hkl_vits_YYYYMMDD_HHMMSS/")
    
    print("\n" + "="*70)
    print("Advanced Options:")
    print("="*70)
    print("\n# Resume from checkpoint:")
    print("  python main.py --resume models/checkpoints/hkl_vits_epoch_50.pt")
    print("\n# Use specific GPU:")
    print("  python main.py --device cuda:0")
    print("\n# Use CPU only:")
    print("  python main.py --device cpu")
    
    response = input("\n\nProceed with training? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        try:
            subprocess.run([sys.executable, 'main.py'], check=False)
        except KeyboardInterrupt:
            print("\n\n[INFO] Training interrupted by user")
    else:
        print("Training cancelled.")


def post_training_guide():
    """Guide for using trained model"""
    print_header("AFTER TRAINING: Using Your Trained Model")
    
    guide = """
1. MODEL LOCATION
   models/final/hkl_vits_YYYYMMDD_HHMMSS/
   - config.json               (Model configuration)
   - model.pt                  (Trained weights)
   - metadata.json             (Model information)
   - README.md                 (Usage instructions)
   - requirements_inference.txt

2. USING IN YOUR PROJECT
   
   Option A: Copy entire model
   cp -r models/final/hkl_vits_* /your/project/models/
   
   Option B: Load checkpoint dynamically
   import torch
   from pathlib import Path
   
   # Load config
   import json
   config_path = Path('models/final/hkl_vits_*/config.json')
   config = json.load(open(config_path))
   
   # Build model
   from hkl_vits.hkl_vits_model import HKLVITS
   model = HKLVITS(
       vocab_size=config['model']['vocab_size'],
       phoneme_vocab=config['model']['phoneme_vocab_size'],
       ...
   )
   
   # Load weights
   checkpoint = torch.load('models/final/hkl_vits_*/model.pt')
   model.load_state_dict(checkpoint['model_state_dict'])

3. INFERENCE
   
   Simple inference:
   from hkl_vits.inference import HKLVITSInference
   
   inference = HKLVITSInference(
       config_path='models/final/hkl_vits_*/config.json',
       checkpoint_path='models/final/hkl_vits_*/model.pt',
       device='cuda'
   )
   
   # Synthesize speech
   text = "ನಮಸ್ಕಾರ, ಇದು ಕನ್ನಡ ಭಾಷಾ ಸಂಶ್ಲೇಷಣ"
   audio = inference.synthesize(text)
   inference.save_audio(audio, 'output.wav')

4. FINE-TUNING FOR NEW SPEAKER
   
   # Load pretrained model
   checkpoint = torch.load('models/final/hkl_vits_*/model.pt')
   model.load_state_dict(checkpoint['model_state_dict'])
   
   # Fine-tune on new speaker (10-100 utterances)
   from training.train_hkl_vits import HKLVITSTrainer
   
   trainer = HKLVITSTrainer('configs/hkl_vits_config.json')
   # ... adapt training loop for few-shot learning

5. DEPLOYMENT OPTIONS
   
   A. Batch Inference (Offline)
      for text_file in text_files:
          audio = inference.synthesize(read_kannada_text(text_file))
          save_audio(audio, output_wav)
   
   B. Real-time Inference (Streaming)
      # Modify inference.py for streaming generation
      stream_output = inference.synthesize_streaming(text, chunk_size=512)
   
   C. Model Quantization (Optimization)
      from torch.quantization import quantize_dynamic
      quantized_model = quantize_dynamic(model, ...)

6. MONITORING & DEBUGGING
   
   Check model size:
   total_params = sum(p.numel() for p in model.parameters())
   trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
   print(f"Total: {total_params/1e6:.1f}M, Trainable: {trainable/1e6:.1f}M")
   
   Inference speed:
   import time
   start = time.time()
   audio = inference.synthesize(text)
   elapsed = time.time() - start
   print(f"Generated {len(audio)/22050:.1f}s speech in {elapsed:.1f}s")
   print(f"Speed: {len(audio)/22050/elapsed:.1f}x realtime")

7. SHARING THE MODEL
   
   Package for distribution:
   tar -czf hkl_vits_kannada_model.tar.gz models/final/hkl_vits_*/
   
   Upload to repository including:
   - model.pt (weights)
   - config.json (configuration)
   - README.md (documentation)
   - metadata.json (information)
   - LICENSE (if applicable)

8. TROUBLESHOOTING
   
   Issue: GPU out of memory
     - Reduce batch size in config
     - Reduce hidden_dim
     - Use gradient accumulation
   
   Issue: Poor audio quality
     - Check training was completed
     - Verify dataset quality
     - Increase training epochs
     - Adjust loss weights
   
   Issue: Model not found
     - Verify models/final/ directory exists
     - Check path format (correct slashes for your OS)
     - Print(Path('models/final/').glob('*')) to list available

9. NEXT STEPS
   
   [TODO] Verify inference works on test Kannada text
   [TODO] Evaluate audio quality (MOS, PESQ, spectral distortion)
   [TODO] Fine-tune for specific domain/speaker if needed
   [TODO] Deploy to production service
   [TODO] Contribute improvements back to project

For detailed documentation, see:
  - docs/UNIQUE_FEATURES_RESEARCH.md (Research contributions)
  - docs/PROJECT_SUMMARY.md (Technical details)
  - docs/IMPLEMENTATION_COMPLETE.md (Implementation status)
"""
    
    print(guide)


def main():
    """Main quick start guide"""
    print("\n" + "="*70)
    print("HKL-VITS: Kannada Text-to-Speech Training Guide")
    print("="*70)
    print("Hybrid Linguistic-Enhanced VITS for Kannada Language")
    print("="*70 + "\n")
    
    # Run verification steps
    checks = [
        ("Environment", verify_environment),
        ("Configuration", check_config),
        ("Data", verify_data),
        ("GPU", check_gpu)
    ]
    
    all_passed = True
    for name, check_func in checks:
        if not check_func():
            all_passed = False
    
    if not all_passed:
        print_header("VERIFICATION FAILED")
        print("Please resolve the issues above before training.")
        sys.exit(1)
    
    print_header("ALL CHECKS PASSED - READY FOR TRAINING")
    
    # Start training
    start_training()
    
    # Show post-training guide
    post_training_guide()


if __name__ == '__main__':
    main()
