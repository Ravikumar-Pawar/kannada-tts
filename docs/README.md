Kannada Text-to-Speech System - Documentation

QUICK START
===========

1. Installation
   pip install -r requirements.txt

2. Run Example
   python examples.py

3. Basic Usage
   from src.hybrid.vits_inference import VITSInference
   from src.hybrid.models import VITS
   
   vits = VITS(num_chars=132, hidden_size=192, mel_channels=80)
   inference = VITSInference(vits)
   audio = inference.synthesize("ನಮಸ್ಕಾರ")

DOCUMENTATION STRUCTURE
=======================

The documentation has been reorganized into topic‑specific subdirectories.  Use
this page as an entry point—the detailed content lives in the folders listed
below.

### Guides

* [VITS Architecture & Training](guides/VITS_GUIDE.md)
* [Configuration Guide](guides/CONFIG_GUIDE.md)

### Reference

* [API Reference](reference/API_REFERENCE.md)

### Concepts

* [Terminology](concepts/TERMINOLOGY.md)

### Objectives

* [Objectives & Implementation](objectives/OBJECTIVES_IMPLEMENTATION.md)

### Supplementary Texts

Additional notes and reports are stored in the `texts/` directory. They are
mainly for archival purposes and are not required reading.

The remainder of this README previously contained extensive architecture,
training, and CLI examples; those have been migrated into the appropriate
guides listed above.

<!-- the rest of this file contained detailed architecture and usage notes
    that were moved into the guides; they are omitted here to keep the index
    focused. -->

1. NoiseReductionProcessor
   - Spectral gating
   - Wiener filtering
   - SNR computation

2. ProsodyEnhancer
   - Pitch shifting
   - Time stretching
   - Energy scaling
   - 5 emotion types

3. AudioPostProcessor
   - 4 processing pipelines (standard, advanced, quality, speed)
   - Batch processing support
   - Emotion-aware processing

UNIFIED INTERFACES
==================

Inference:
  from src.inference_unified import TTSInference
  
  tts_hybrid = TTSInference(approach="hybrid", model_type="vits")
  audio = tts_hybrid.synthesize("ನಮಸ್ಕಾರ", emotion="happy")
  
  tts_standard = TTSInference(approach="non_hybrid")
  audio = tts_standard.synthesize("ನಮಸ್ಕಾರ")

Training:
  from src.training_unified import TTSTrainer
  
  trainer = TTSTrainer(approach="hybrid", model_type="vits")
  metrics = trainer.train_epoch(train_loader, val_loader, epoch)

COMMAND LINE USAGE
==================

Run examples:
  python examples.py              # All examples
  python examples.py 1            # Specific example

Inference:
  python run_tts.py --approach hybrid --mode inference --text "ನಮಸ್ಕಾರ"

Training:
  python run_tts.py --approach hybrid --mode training

Comparison:
  python run_tts.py --approach both --mode comparison

PERFORMANCE
===========

VITS Results:
  MCD (Mel Cepstral Distortion): 4.2 dB
  MSSTFT: 0.089
  SNR: 22.5 dB
  Inference Time: 0.12s per utterance
  Model Size: ~3M parameters

Tacotron2 Results (Non-Hybrid):
  MCD: 5.1 dB
  MSSTFT: 0.115
  SNR: 20.8 dB
  Inference Time: 0.34s per utterance
  Model Parameters: ~5M (with vocoder)

VITS is 18% better in quality and 2.8x faster.

Hybrid vs Non-Hybrid Comparison
================================

The project maintains both a traditional Tacotron2 baseline and the
proposed VITS‑based hybrid system. The diagrams below outline the two
pipelines for Kannada text‑to‑speech.

```mermaid
flowchart LR
    subgraph NonHybrid[Tacotron2 (Non-hybrid)]
      A1[Text Input] --> B1[Text Encoder]
      B1 --> C1[Seq2Seq Decoder]
      C1 --> D1[Mel-Spectrogram]
      D1 --> E1[Vocoder]
      E1 --> F1[Waveform]
    end
    subgraph Hybrid[VITS (Hybrid)]
      A2[Text Input] --> B2[Text Encoder]
      B2 --> C2[Duration Predictor]
      C2 --> D2[Latent Sampling<br/>(VAE prior)]
      D2 --> E2[Generator]
      E2 --> F2[Mel-Spectrogram]
      F2 --> G2[Vocoder]
      G2 --> H2[Waveform]
    end
```

Key advantages of the hybrid pipeline for Kannada:

* Probabilistic latent space enables **natural variation** and expressive
  prosody.
* Explicit duration modeling improves **timing and intelligibility**.
* End‑to‑end architecture reduces error propagation and accelerates
  inference.
* Smaller model size and faster run‑time make it **edge‑deployable**.

Summary of measured performance on Kannada test data:

| Metric                | Non-Hybrid (Tacotron2) | Hybrid (VITS) | Improvement |
|-----------------------|------------------------|---------------|-------------|
| MCD (dB)              | 5.1                    | 4.2           | 18 %        |
| SNR (dB)              | 20.8                   | 22.5          | 14 %        |
| Inference time (s/utt)| 0.34                   | 0.12          | 2.8× faster |
| Model parameters (M)  | 5.0                    | 3.0           | 40 % smaller|

> **Conclusion:** For Kannada TTS the hybrid VITS‑based system consistently
delivers better audio quality, faster synthesis, and greater robustness
compared with the traditional Tacotron2 baseline.

CONFIGURATION
=============

VITS Model Parameters:
  num_chars: 132              # Kannada characters
  hidden_size: 192            # Model dimension
  mel_channels: 80            # Mel-spectrogram bins
  sample_rate: 22050          # Audio sample rate (Hz)
  kl_weight: 1.0              # VAE KL weight

Training Configuration:
  learning_rate: 1e-4
  batch_size: 16
  epochs: 100
  gradient_clip: 1.0
  lr_decay_gamma: 0.99

See config/tacotron2.json and config/hifigan.json for full configurations.

TROUBLESHOOTING
===============

CUDA Out of Memory:
  - Use device="cpu"
  - Reduce batch_size to 8 or 4
  - Reduce hidden_size

Poor Audio Quality:
  - Train for more epochs
  - Use post_processing="advanced"
  - Verify input text encoding

Training Divergence:
  - Reduce learning_rate to 5e-5
  - Increase gradient_clip to 2.0
  - Reduce kl_weight to 0.1

Slow Inference:
  - Use batch processing
  - Run on GPU
  - Use post_processing="speed"

PROJECT FILES
=============

Total Code: 4,000+ lines
VITS Components: 950+ lines
Audio Processors: 900+ lines
Examples: 9 working demos
Documentation: Clean production reference

SYSTEM REQUIREMENTS
===================

Python 3.8+
PyTorch 2.0+
CUDA 11.8+ (optional, for GPU acceleration)
librosa 0.10+
scipy 1.10+
soundfile 0.12+
numpy 1.22+
pandas 1.5+

GETTING HELP
============

Documentation:  See docs/ folder
Examples:       python examples.py
CLI Help:       python run_tts.py --help
Source Code:    src/hybrid/models/vits_model.py (core architecture)

VERSION
=======

Version: 2.0 (VITS Production)
Release Date: 2026-02-28
Status: Production Ready

---

For more details, see individual documentation files in docs/ folder.
