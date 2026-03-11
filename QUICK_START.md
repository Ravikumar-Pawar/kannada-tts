# QUICK_START.md

# HKL-VITS Quick Start Guide

## Installation & Setup (5 minutes)

```bash
# 1. Clone and navigate
cd kannada-hkl-vits

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python -c "import torch, torchaudio; print('✓ Dependencies OK')"
```

## Prepare Dataset (10 minutes)

```bash
# Directory structure required:
# dataset/
# ├── wav/
# │   ├── sample_001.wav
# │   ├── sample_002.wav
# │   └── ...
# └── txt/
#     ├── sample_001.txt
#     ├── sample_002.txt
#     └── ...

# Validate dataset
python hkl_vits/utils.py prepare_dataset --data_dir path/to/dataset

# Split into train/val/test
python hkl_vits/utils.py split_dataset --data_dir path/to/dataset
```

## Train Model (hours to days, depending on GPU)

```bash
# Basic training
python training/train_hkl_vits.py \
    --config configs/hkl_vits_config.json \
    --data_dir path/to/dataset \
    --gpu 0

# Monitor training
# Logs appear in: logs/training_YYYYMMDD_HHMMSS.log
# Checkpoints saved to: checkpoints/hkl_vits_epoch_X.pt

# Resume training from checkpoint (if interrupted)
python training/train_hkl_vits.py \
    --config configs/hkl_vits_config.json \
    --data_dir path/to/dataset \
    --resume checkpoints/hkl_vits_epoch_50.pt \
    --gpu 0
```

## Test Inference (immediate)

```bash
# Single text synthesis
python hkl_vits/inference.py \
    --config configs/hkl_vits_config.json \
    --checkpoint checkpoints/hkl_vits_epoch_100.pt \
    --text "ನಮಸ್ತೆ ಪ್ರಪಂಚ" \
    --output output.wav

# Interactive synthesis
python hkl_vits/inference.py \
    --config configs/hkl_vits_config.json \
    --checkpoint checkpoints/hkl_vits_epoch_100.pt \
    --interactive

# In interactive mode, type Kannada text and press Enter
# Commands: p PITCH, e ENERGY, l LENGTH, t TEMP, q to quit
```

## Evaluate Model

```bash
python training/evaluate.py \
    --config configs/hkl_vits_config.json \
    --checkpoint checkpoints/hkl_vits_epoch_100.pt \
    --data_dir path/to/test/data \
    --output evaluation_report.json
```

---

## File Organization

```
Created/Modified Files:
├── configs/hkl_vits_config.json           ← Complete 80+ parameter config
├── hkl_vits/
│   ├── __init__.py                         ← Package init
│   ├── grapheme_encoder.py                 ← Transformer (4-layer, 4-head)
│   ├── phoneme_encoder.py                  ← BiLSTM (2-layer, bidirectional)
│   ├── fusion_layer.py                     ← Multiple fusion strategies
│   ├── prosody_encoder.py                  ← F0 + Energy conditioning
│   ├── hkl_vits_model.py                   ← Main VITS architecture
│   ├── kannada_g2p.py                      ← Kannada phoneme converter
│   ├── dataset_loader.py                   ← Data pipeline + prosody extraction
│   ├── loss_functions.py                   ← 5-term multi-objective loss
│   ├── inference.py                        ← Inference + interactive mode
│   └── utils.py                            ← Dataset preparation utilities
├── training/
│   ├── train_hkl_vits.py                   ← Full training pipeline
│   └── evaluate.py                         ← Evaluation metrics (MCD, PESQ)
├── requirements.txt                        ← All dependencies
├── README.md                               ← Comprehensive documentation
├── PROJECT_SUMMARY.md                      ← Technical deep-dive
└── QUICK_START.md                          ← This file
```

---

## Key Commands Reference

| Task | Command |
|------|---------|
| **Prepare Data** | `python hkl_vits/utils.py prepare_dataset --data_dir DATA` |
| **Train** | `python training/train_hkl_vits.py --config CONFIG --data_dir DATA --gpu 0` |
| **Resume Training** | `python training/train_hkl_vits.py --config CONFIG --data_dir DATA --resume CKPT --gpu 0` |
| **Synthesize** | `python hkl_vits/inference.py --config CONFIG --checkpoint CKPT --text TEXT --output WAV` |
| **Interactive** | `python hkl_vits/inference.py --config CONFIG --checkpoint CKPT --interactive` |
| **Evaluate** | `python training/evaluate.py --config CONFIG --checkpoint CKPT --data_dir DATA` |
| **Print Config** | `python hkl_vits/utils.py print_config --config CONFIG` |

---

## Python API Usage

```python
# Training
from training.train_hkl_vits import HKLVITSTrainer

trainer = HKLVITSTrainer('configs/hkl_vits_config.json', device='cuda')
trainer.train(data_dir='path/to/dataset', num_epochs=100)

# Inference
from hkl_vits.inference import HKLVITSInference

inference = HKLVITSInference(
    config_path='configs/hkl_vits_config.json',
    checkpoint_path='checkpoints/hkl_vits_epoch_100.pt'
)

waveform = inference.synthesize(
    kannada_text='ನಮಸ್ತೆ',
    save_path='output.wav'
)

# Batch synthesis
texts = ['ಹಲೋ', 'ಧನ್ಯವಾದ', 'ನಮಸ್ತೆ']
waveforms = inference.synthesize_batch(texts, save_dir='outputs')

# Evaluation
from training.evaluate import HKLVITSEvaluator
from hkl_vits.dataset_loader import get_dataloaders
from hkl_vits.loss_functions import HKLVITSLoss

evaluator = HKLVITSEvaluator(
    config_path='configs/hkl_vits_config.json',
    checkpoint_path='checkpoints/hkl_vits_epoch_100.pt'
)

val_loader, _ = get_dataloaders('path/to/data', batch_size=32)
criterion = HKLVITSLoss()
report = evaluator.generate_report(val_loader, criterion, save_path='report.json')
```

---

## Configuration Customization

Edit `configs/hkl_vits_config.json` to adjust:

```json
{
  "training": {
    "batch_size": 32,           // Reduce if OOM
    "learning_rate": 0.0002,    // Learning rate
    "num_epochs": 100,          // Training duration
    "warmup_steps": 5000        // Warmup iterations
  },
  "loss_weights": {
    "reconstruction": 1.0,      // Mel-spec accuracy
    "kl_divergence": 0.1,       // Latent regularization
    "adversarial": 1.0,         // GAN loss
    "f0": 0.5,                  // Pitch accuracy
    "energy": 0.1               // Energy accuracy
  },
  "model": {
    "hidden_dim": 256,          // Embedding dimension
    "num_mels": 80,             // Mel-spectrogram bins
    "sample_rate": 22050        // Hz
  }
}
```

---

## Troubleshooting

### Error: "CUDA out of memory"
```bash
# In config, reduce:
# - batch_size (try 16 or 8)
# - max_audio_length
# Or use CPU: --gpu -1
```

### Error: "No module named torch"
```bash
pip install -r requirements.txt
```

### Error: "Text file not found"
```bash
# Ensure dataset structure:
dataset/
├── wav/  ← audio files
└── txt/  ← text files (same stems)
```

### Poor quality output
```bash
# Check:
1. Dataset quality (no background noise)
2. Training convergence (check loss curves)
3. Number of epochs (try 200+)
4. Loss weights (tune if needed)
```

---

## Performance Tips

1. **Faster Training**
   - Use multiple GPUs (DDP if available)
   - Increase batch size (if VRAM allows)
   - Use mixed precision (set in config)

2. **Better Quality**
   - Train longer (200+ epochs)
   - Use more data (1000+ samples)
   - Tune loss weights
   - Use better quality audio

3. **Faster Inference**
   - Use GPU: `--gpu 0`
   - Batch synthesis for multiple texts
   - Consider model quantization

---

## Sample Kannada Texts for Testing

```
ನಮಸ್ತೆ                          (Namaste)
ಹೊಮ್ಮೆ ಬರಾ                       (Please come)
ನಿನ್ನ ಹೆಸರೇನು                    (What is your name)
ಉದ್ಯಮದಿಂದ ದೋಸ್ತಿ ಸಿಕ್ಕುವುದಿಲ್ಲ    (Friendship doesn't come by effort)
```

---

## Next Steps

1. ✅ Review `README.md` for detailed documentation
2. ✅ Check `PROJECT_SUMMARY.md` for technical details
3. ✅ Prepare your Kannada dataset in required format
4. ✅ Train the model on your data
5. ✅ Evaluate and fine-tune
6. ✅ Deploy for production use

---

## Support & Resources

- **Documentation**: See `README.md`
- **Technical Details**: See `PROJECT_SUMMARY.md`
- **Issues**: Check logs in `logs/` directory
- **Configuration**: Edit `configs/hkl_vits_config.json`

---

## Project Status

✅ **COMPLETE AND READY FOR USE**

All components implemented, tested, and documented.

---

*Last Updated: March 11, 2026*
