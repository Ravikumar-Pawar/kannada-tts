# Configuration Guide - Kannada TTS System

## Overview
This guide explains all configuration parameters and how to tune them for your specific needs.

---

## Tacotron2 Configuration (`config/tacotron2.json`)

### Basic Settings
```json
{
  "model": "tacotron2",
  "run_name": "kannada_tts_baseline",
  "epochs": 500,
  "batch_size": 16,
  "eval_batch_size": 8
}
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `epochs` | 500 | Total training epochs |
| `batch_size` | 16 | Training batch size (reduce if OOM) |
| `eval_batch_size` | 8 | Evaluation batch size |
| `run_eval` | true | Evaluate during training |
| `save_step` | 1000 | Save model every N steps |
| `eval_steps` | 500 | Evaluate every N steps |

### Learning Rate Configuration
```json
{
  "lr": 0.001,
  "lr_scheduler": "Noam",
  "lr_scheduler_params": {
    "warmup_steps": 4000
  }
}
```

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `lr` | 0.001 | Initial learning rate |
| `lr_scheduler` | "Noam" | Noam/Transformer scheduler |
| `warmup_steps` | 4000 | Steps to warmup |

### Audio Configuration
```json
{
  "audio": {
    "sample_rate": 22050,
    "hop_length": 256,
    "win_length": 1024,
    "n_mel_channels": 80,
    "mel_fmin": 0,
    "mel_fmax": 8000
  }
}
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `sample_rate` | 22050 | Audio sample rate (Hz) |
| `hop_length` | 256 | STFT hop length |
| `win_length` | 1024 | STFT window length |
| `n_mel_channels` | 80 | Mel-spectrogram bins |
| `mel_fmin` | 0 | Min mel frequency (Hz) |
| `mel_fmax` | 8000 | Max mel frequency (Hz) |

### Model Architecture
```json
{
  "model_args": {
    "encoder_hidden_size": 256,
    "encoder_num_convolutions": 3,
    "encoder_conv_filters": 512,
    "decoder_hidden_size": 1024,
    "decoder_lstm_layers": 2,
    "attention_hidden_size": 128,
    "postnet_num_convolutions": 5
  }
}
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `encoder_hidden_size` | 256 | Encoder hidden dimension |
| `decoder_hidden_size` | 1024 | Decoder hidden dimension |
| `decoder_lstm_layers` | 2 | Number of LSTM layers |
| `attention_hidden_size` | 128 | Attention mechanism size |

### Character Set
```json
{
  "characters": {
    "pad": "_",
    "eos": "~",
    "bos": "^",
    "blank": " ",
    "characters": "!'.(),-.:;?ಅಆಇಈಉಊಋಎಏಐಒಓಔಕಖಗಘಙಚಛಜಝಞಟಠಡಢಣತಥದಧನಪಫಬಭಮಯರಲವಶषಸಹೃೈೊೋೌಂಃೞ"
  }
}
```

- 132 Kannada characters (U+0C80 - U+0CFF range)
- Diacritics and sign marks included
- Common punctuation marks

---

## HiFiGAN Configuration (`config/hifigan.json`)

```json
{
  "model": "hifigan",
  "run_name": "kannada_hifigan",
  "epochs": 200,
  "batch_size": 16,
  "eval_batch_size": 8,
  "num_loader_workers": 4,
  "audio": {
    "sample_rate": 22050,
    "hop_length": 256
  },
  "lr": 0.0002,
  "save_step": 5000
}
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `epochs` | 200 | Training epochs for vocoder |
| `lr` | 0.0002 | Learning rate |
| `save_step` | 5000 | Save frequency |

---

## Tuning Guidelines

### For Better Quality
```json
{
  "epochs": 1000,              // Increase training
  "batch_size": 32,            // Larger batches
  "encoder_hidden_size": 512,  // Larger model
  "decoder_hidden_size": 2048, //
  "n_mel_channels": 128,       // More mel bins
  "lr": 0.0005                 // Slower learning
}
```

### For Faster Training
```json
{
  "epochs": 100,               // Fewer epochs
  "batch_size": 8,             // Smaller batches
  "encoder_hidden_size": 128,  // Smaller model
  "decoder_hidden_size": 512,  //
  "n_mel_channels": 40,        // Fewer mel bins
  "save_step": 5000            // Less frequent saves
}
```

### For GPU with Limited Memory
```json
{
  "batch_size": 4,             // Small batches
  "eval_batch_size": 2,        //
  "num_loader_workers": 0,     // Reduce workers
  "encoder_hidden_size": 128,  // Smaller model
  "decoder_hidden_size": 256,  //
  "save_n_checkpoints": 1      // Keep only 1 checkpoint
}
```

### For CPU Training (Not Recommended)
```json
{
  "epochs": 50,                // Very few epochs
  "batch_size": 2,             // Minimal batch
  "num_loader_workers": 0,     // No multiprocessing
  "encoder_hidden_size": 64,   // Minimal model
  "decoder_hidden_size": 256,  //
  "n_mel_channels": 40         //
}
```

---

## Noise Reduction Configuration

In `src/inference.py`, adjust noise reduction parameters:

```python
# Spectral Gating Threshold
threshold_db = -40  # Lower = more aggressive denoising
                    # Range: -60 to -20 dB

# Wiener Filter Noise Profile Duration
noise_profile_duration = 0.5  # Seconds (typical: 0.1-1.0)
```

### Noise Reduction Methods
- **spectral_gating**: Fast, good for background noise
- **wiener**: More sophisticated, better for varying noise

---

## Emotion Enhancement Presets

Customize in `src/inference.py`:

```python
emotion_params = {
    "happy": {
        "pitch": 2.0,        # Semitones
        "duration": 0.9,     # Speed multiplier
        "energy": 1.2        // Amplitude multiplier
    },
    "sad": {
        "pitch": -1.5,
        "duration": 1.2,
        "energy": 0.8
    }
    # ... etc
}
```

---

## Performance Tuning

### For Real-Time Inference
- Use smaller model (`encoder_hidden_size: 128`)
- Set `batch_size: 1`
- Run on GPU if available

### For Maximum Quality
- Increase training epochs to 1000+
- Use larger model (double hidden sizes)
- Use HiFiGAN vocoder
- Increase `n_mel_channels` to 128

### For Batch Processing
- Use larger `batch_size` (32-64)
- Enable multi-GPU if available
- Use parallel workers: `num_loader_workers: 8`

---

## Troubleshooting Configuration

### CUDA Out of Memory
```python
# In config
"batch_size": 4        # Reduce
"eval_batch_size": 2
"num_loader_workers": 0  # Disable
```

### Slow Training
```python
# Check:
# - GPU utilization (should be > 80%)
# - Increase num_loader_workers
# - Increase batch_size (if memory allows)
# - Check for I/O bottlenecks
```

### Poor Audio Quality
```python
# Increase:
# - epochs (more training)
# - n_mel_channels (128 instead of 80)
# - model size (double hidden sizes)
# - Decrease lr_scheduler warmup_steps (faster ramp-up)
```

### Training Divergence
```python
# Reduce:
# - lr (try 0.0005)
# - batch_size
# - Increase warmup_steps
```

---

## Advanced Settings

### Custom Character Set
Modify `characters` field to include/exclude characters:

```json
"characters": "!'.(),-.:;?ಅಆಇ..."  // Add/remove Kannada chars
```

### Phoneme Support
To enable phoneme-based training:

```json
"use_phonemes": true,
"phoneme_language": "kn"  // Kannada
```

### Text Cleaner
```json
"text_cleaner": "basic_cleaners"  // or "multilingual_cleaners"
```

---

## Recommended Configurations by Hardware

### NVIDIA RTX 4090 (24GB)
```json
{
  "batch_size": 64,
  "encoder_hidden_size": 512,
  "decoder_hidden_size": 2048,
  "epochs": 1000,
  "num_loader_workers": 8
}
```

### NVIDIA T4 (16GB)
```json
{
  "batch_size": 32,
  "encoder_hidden_size": 256,
  "decoder_hidden_size": 1024,
  "epochs": 500,
  "num_loader_workers": 4
}
```

### NVIDIA GTX 1080 (8GB)
```json
{
  "batch_size": 16,
  "encoder_hidden_size": 256,
  "decoder_hidden_size": 512,
  "epochs": 300,
  "num_loader_workers": 2
}
```

### CPU Only
```json
{
  "batch_size": 2,
  "encoder_hidden_size": 128,
  "decoder_hidden_size": 256,
  "epochs": 100,
  "num_loader_workers": 0
}
```

---

## Monitoring Configuration

### TensorBoard Logging
```bash
# During training
tensorboard --logdir output/tacotron2/
```

### Logging Output
- Console: Real-time updates
- File: `output/training.log`
- TensorBoard: `output/tacotron2/runs/`

---

## Validation Configuration

Adjust in data_prep.py:

```python
# Audio validation thresholds
MIN_DURATION = 1.0   # seconds
MAX_DURATION = 30.0
TARGET_SR = 22050    # Hz

# Text validation
MIN_TEXT_LENGTH = 3  # characters
```

---

## Final Tips

✅ Start with default config  
✅ Monitor training metrics  
✅ Adjust if converging too slow/diverging  
✅ Save configs before experimentation  
✅ Use validation set to track improvements  
✅ Keep backup of best models  
