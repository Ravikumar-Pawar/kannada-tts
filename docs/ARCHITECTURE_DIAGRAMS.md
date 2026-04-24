# HKL-VITS Architecture Diagrams

## 1. HIGH-LEVEL ARCHITECTURE

### System Overview: From Kannada Text to Speech

```
┌─────────────────────────────────────────────────────────────────┐
│                  KANNADA TEXT INPUT                             │
│              "ನಮಸ್ಕಾರ, ನೀವು ಸುಖವಿದ್ದೀರಿ?"                 │
└─────────────────────┬───────────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
   ┌─────────────┐          ┌──────────────┐
   │  GRAPHEME   │          │   PHONEME    │
   │  ENCODER    │          │   ENCODER    │
   │ (Transform) │          │   (BiLSTM)   │
   └──────┬──────┘          └──────┬───────┘
          │                        │
          └────────────┬───────────┘
                       │
                       ▼
            ┌────────────────────┐
            │  FUSION LAYER      │
            │ (Multi-head Attn)  │
            └─────────┬──────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │  PROSODY ENCODER       │
         │  - Pitch (F0)          │
         │  - Energy              │
         └────────────┬───────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │ POSTERIOR ENCODER      │
         │ (Mel-spectrogram)      │
         └────────────┬───────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │  FLOW-BASED LATENT     │
         │  MODEL (4 flows)       │
         └────────────┬───────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │   HiFi-GAN DECODER     │
         │   (Neural Vocoder)     │
         └────────────┬───────────┘
                      │
                      ▼
        ┌──────────────────────────┐
        │   WAVEFORM OUTPUT        │
        │  (22.05 kHz Audio)       │
        └──────────────────────────┘
```

---

## 2. LOW-LEVEL ARCHITECTURE

### Detailed Component Breakdown

#### A. DUAL ENCODER SYSTEM

```
GRAPHEME ENCODER (Transformer-based)
================================

Input Text: ನಮSample
    │
    ▼
[EMBEDDING LAYER]
    │ (256-dim embeddings)
    ▼
┌──────────────────────────┐
│ TRANSFORMER BLOCK 1      │
│ - Multi-head Attention   │
│ - Feed-forward Network   │
│ - Layer Norm             │
│ (4 heads, 256 hidden)    │
└─────────┬────────────────┘
          │
          ▼
┌──────────────────────────┐
│ TRANSFORMER BLOCK 2      │
│ - Multi-head Attention   │
│ - Feed-forward Network   │
│ - Layer Norm             │
└─────────┬────────────────┘
          │
          ▼
┌──────────────────────────┐
│ TRANSFORMER BLOCK 3      │
│ - Multi-head Attention   │
│ - Feed-forward Network   │
│ - Layer Norm             │
└─────────┬────────────────┘
          │
          ▼
┌──────────────────────────┐
│ TRANSFORMER BLOCK 4      │
│ - Multi-head Attention   │
│ - Feed-forward Network   │
│ - Layer Norm             │
└─────────┬────────────────┘
          │
          ▼
Grapheme Context: [seq_len, 256]


PHONEME ENCODER (BiLSTM-based)
==============================

Input Phonemes: /n a m a s k a: r a/
    │
    ▼
[EMBEDDING LAYER]
    │ (256-dim embeddings)
    ▼
┌───────────────────────────────┐
│ BI-LSTM LAYER 1               │
│ Forward:  →→→→→→→→→→         │
│ Backward: ←←←←←←←←←←         │
│ Hidden: 256                   │
└─────────────┬─────────────────┘
              │
              ▼
┌───────────────────────────────┐
│ BI-LSTM LAYER 2               │
│ Forward:  →→→→→→→→→→         │
│ Backward: ←←←←←←←←←←         │
│ Hidden: 256                   │
└─────────────┬─────────────────┘
              │
              ▼
Phoneme Context: [seq_len, 256]
```

#### B. FUSION LAYER

```
Grapheme Context [seq_len, 256]    Phoneme Context [seq_len, 256]
         │                                  │
         └──────────────┬───────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │  MULTI-HEAD ATTENTION         │
        │                               │
        │  Head 1: Query=Grapheme       │
        │          Key=Phoneme          │
        │          Value=Phoneme        │
        │  Head 2: Similar...           │
        │  Head 3: Similar...           │
        │  Head 4: Similar...           │
        │                               │
        │  Output: Attention Weights    │
        └──────────────┬────────────────┘
                       │
                       ▼
        ┌──────────────────────────┐
        │  LINEAR PROJECTION       │
        │  [256] -> [256]          │
        └──────────────┬───────────┘
                       │
                       ▼
        ┌──────────────────────────┐
        │  RESIDUAL CONNECTION     │
        │  Fused = Grapheme +      │
        │          Projected       │
        └──────────────┬───────────┘
                       │
                       ▼
        Fused Representation: [seq_len, 256]
```

#### C. PROSODY ENCODER

```
Audio Features (from Mel-spectrogram)
         │
    ┌────┴───┐
    │         │
    ▼         ▼
┌────────┐ ┌────────┐
│  PITCH │ │ENERGY  │
│ (F0)   │ │        │
└────┬───┘ └───┬────┘
     │         │
     ▼         ▼
┌──────────────────────────┐
│  QUANTIZATION BINS       │
│  - 256 pitch bins        │
│    (f0_min=70 to        │
│     f0_max=400 Hz)       │
│  - 256 energy bins       │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  EMBEDDING LAYERS        │
│  - Pitch embedding [256] │
│  - Energy embedding [256]│
└──────────┬───────────────┘
           │
           ▼
Prosody Features: [seq_len, 256]
```

#### D. POSTERIOR ENCODER & FLOW MODEL

```
Mel-spectrogram Input
[batch, 80 mels, time]
         │
         ▼
    ┌─────────────┐
    │ Conv1d Pre  │
    │ [80] -> [256]
    └──────┬──────┘
           │
           ▼
    ┌──────────────────┐
    │ Conv Block 1     │
    │ - Conv(3x3)      │
    │ - ReLU           │
    │ - Conv(3x3)      │
    │ - ReLU           │
    │ + Skip Connection│
    └──────┬───────────┘
           │
      [Repeat 3x]
           │
           ▼
    ┌──────────────────┐
    │ Conv1d Post      │
    │ [256] -> [512]   │
    │ Split into:      │
    │ - mu [256]       │
    │ - log_var [256]  │
    └──────┬───────────┘
           │
           ▼
    ┌─────────────────────────────┐
    │  FLOW TRANSFORMATION        │
    │                             │
    │  Flow 1: Affine coupling    │
    │  Flow 2: Affine coupling    │
    │  Flow 3: Affine coupling    │
    │  Flow 4: Affine coupling    │
    │                             │
    │  Each flow:                 │
    │  - Splits channels          │
    │  - Transforms one half      │
    │  - Couples with other half  │
    │  - Invertible transformation
    └──────┬──────────────────────┘
           │
           ▼
    Latent Distribution
    [batch, 256, time]
```

#### E. HiFi-GAN DECODER

```
Latent Vector [batch, 256, time]
         │
         ▼
    ┌─────────────┐
    │ Pre-Conv    │
    │ [256] -> [512]
    └──────┬──────┘
           │
           ▼
    ┌──────────────────────┐
    │ UPSAMPLE 1: x8       │
    │ Conv(512, 512, 16)   │
    │ + ReLU               │
    │ [seq] -> [seq*8]     │
    └──────┬───────────────┘
           │
           ▼
    ┌──────────────────────┐
    │ MRD BLOCK 1          │
    │ Multi-Receptive-Dil. │
    │ (kernels: 3,7,11)    │
    │ (dilations: 1,3,5)   │
    └──────┬───────────────┘
           │
      [Repeat Upsample + MRD]
      (3 more times)
           │
           ▼
    ┌──────────────────────┐
    │ Post-Conv            │
    │ [512] -> [1]         │
    │ Tanh activation      │
    └──────┬───────────────┘
           │
           ▼
    Waveform Output
    [batch, 1, time_steps]
    (22.05 kHz audio)
```

---

## 3. DATA FLOW DURING TRAINING

```
┌──────────────────────────────────┐
│   TRAINING ITERATION             │
│                                  │
│  Input:                          │
│  - Kannada text                  │
│  - Target mel-spectrogram        │
│  - Target pitch (F0)             │
│  - Target energy                 │
└────────────┬─────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
┌─────────────┐  ┌──────────────┐
│  FORWARD    │  │  EXTRACTION  │
│   PASS      │  │              │
│             │  │ From target: │
│  Generate   │  │ - Phonemes   │
│  - Grapheme │  │ - Pitch      │
│  - Phoneme  │  │ - Energy     │
│  - Fused    │  │ - Mel-spec   │
│  - Prosody  │  │              │
│  - Posterior│  │              │
│  - Flow     │  │              │
│  - Audio    │  │              │
└─────────┬───┘  └──────┬───────┘
          │             │
          └──────┬──────┘
                 │
                 ▼
        ┌────────────────────┐
        │  LOSS COMPUTATION  │
        │                    │
        │  L_recon = MSE     │
        │  L_kl = KL Div     │
        │  L_f0 = Pitch loss │
        │  L_energy = Energy │
        │  L_adv = Adversarial
        │  L_phoneme = Clarity
        │                    │
        │  Total Loss =      │
        │  λ1*L_recon +      │
        │  λ2*L_kl +         │
        │  λ3*L_f0 +         │
        │  λ4*L_energy +     │
        │  λ5*L_adv +        │
        │  λ6*L_phoneme      │
        └────────┬───────────┘
                 │
                 ▼
        ┌────────────────────┐
        │  BACKWARD PASS     │
        │                    │
        │  Compute gradients │
        │  for all params    │
        └────────┬───────────┘
                 │
                 ▼
        ┌────────────────────┐
        │  GRADIENT CLIPPING │
        │  clip_norm = 1.0   │
        └────────┬───────────┘
                 │
                 ▼
        ┌────────────────────┐
        │  OPTIMIZER STEP    │
        │  Adam with:        │
        │  - lr: 2e-4        │
        │  - betas: (0.9,0.999)
        │  - eps: 1e-8       │
        │  - weight_decay    │
        └────────┬───────────┘
                 │
                 ▼
        ┌────────────────────┐
        │  SCHEDULER STEP    │
        │  ExponentialLR     │
        │  gamma: 0.9999     │
        └────────┬───────────┘
                 │
                 ▼
        Update model parameters
        for next iteration
```

---

## 4. INFERENCE FLOW (SIMPLIFIED)

```
Kannada Text Input
    "ನಮಸ್ಕಾರ"
         │
         ▼
    ┌─────────────────┐
    │ GRAPHEME ENCODE │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ G2P CONVERSION  │
    │ (Kannada-specific)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ PHONEME ENCODE  │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ FUSION          │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ SAMPLE Z from   │
    │ Latent space    │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ INVERSE FLOW    │
    │ Transform Z     │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ HIFI-GAN DECODE │
    │ to Waveform     │
    └────────┬────────┘
             │
             ▼
    Output Audio
    (22.05 kHz .wav)
```

---

## 5. PARAMETER DISTRIBUTIONS

```
MODEL PARAMETERS BREAKDOWN
===========================

Total: ~120 Million Parameters

├─ Grapheme Encoder:     ~15M (12%)
│  ├─ Embeddings:        2M
│  ├─ 4x Transformer:    13M
│  └─ Output projection: 0.3M
│
├─ Phoneme Encoder:      ~5M (4%)
│  ├─ Embeddings:        1.5M
│  ├─ 2x BiLSTM:         3M
│  └─ Output projection: 0.5M
│
├─ Fusion Layer:         ~0.25M (0.2%)
│  ├─ Multi-head Attn:   0.2M
│  └─ Linear:            0.05M
│
├─ Prosody Encoder:      ~3M (2.5%)
│  ├─ Pitch Embedding:   1.5M
│  ├─ Energy Embedding:  1.5M
│  └─ Projection:        0.1M
│
├─ Posterior Encoder:    ~15M (12%)
│  ├─ Pre-Conv:          0.5M
│  ├─ 4x Conv Blocks:    14M
│  └─ Post-Conv:         0.5M
│
├─ Flow Model:           ~8M (7%)
│  └─ 4x Flow Blocks:    8M (per block: 2M)
│
├─ HiFi-GAN Decoder:     ~73M (61%)
│  ├─ Pre-Conv:          0.5M
│  ├─ Upsample Layers:   20M
│  ├─ MRD Blocks:        52M
│  └─ Post-Conv:         0.5M
│
└─ Discriminators:       ~1.5M (1.3%)
   ├─ MPD (Multi-Period):   0.7M
   └─ MSD (Multi-Scale):    0.8M

Model Size: ~350-400 MB (FP32)
```

---

## 6. KANNADA-SPECIFIC PROCESSING

```
INPUT TEXT: "ಸ್ಮರಿಸುತ್ತೇನೆ"
         │
         ▼
GRAPHEME BREAKUP: ಸ್ | ಮ | ರ | ಿ | ಸು | ತ್ | ತೇ | ನೆ
         │
         ▼
G2P CONVERSION (Kannada Rules):
┌──────────────────────────────────────┐
│ s + ್ (virama) + m = sm (cluster)    │
│ → /sma/ (phonetic output)            │
│                                      │
│ Check vowel markers:                 │
│ ಸುತ್ತೇನೆ has ು (u) and ೇ (e:)     │
│ → preserves vowel length             │
│                                      │
│ Gemination check:                    │
│ ಠ್ರ or ತ್ತ = double consonant     │
│ → longer duration in prosody         │
│                                      │
│ Anusvara/Visarga:                    │
│ ನ್ = /n/, ಹ್ = /h/, etc.            │
└──────────────────────────────────────┘
         │
         ▼
OUTPUT PHONEMES:
/s/ + /m/ + /a/ + /r/ + /i/ + /s/ + /u/ + /t:/ + /e:/ + /n/ + /e/
         │
         ▼
PROSODY EXTRACTION:
├─ F0 (Pitch):
│  ├─ /s/ = 150 Hz
│  ├─ /m/ = 155 Hz
│  ├─ /a/ = 180 Hz (peak vowel)
│  ├─ /r/ = 160 Hz
│  └─ ...
│
└─ Energy:
   ├─ /s/ = low (fricative)
   ├─ /m/ = high (nasal with closure)
   └─ /a/ = highest (vowel)
```

---

## 7. TRAINING PIPELINE ARCHITECTURE

```
DATA LOADING PHASE
┌─────────────────────────────────────┐
│                                     │
│  Raw Dataset:                       │
│  data/kannada_tts_dataset/          │
│  ├─ wav/  (audio files)             │
│  ├─ txt/  (transcriptions)          │
│  └─ metadata.tsv                    │
│                                     │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  DATA LOADER (parallel, num_worker) │
│                                     │
│  For each sample:                   │
│  1. Load .wav file                  │
│  2. Load corresponding .txt         │
│  3. Resample to 22.05 kHz           │
│  4. Compute MEL-spectrogram         │
│  5. Extract F0 (pitch)              │
│  6. Extract Energy                  │
│  7. Trim silence                    │
│  8. Padding/Truncation              │
│                                     │
└────────────┬────────────────────────┘
             │
      [Batch Size: 32]
             │
             ▼
TRAINING LOOP (100 epochs)
┌─────────────────────────────────────┐
│  for epoch in range(num_epochs):    │
│    for batch in train_loader:       │
│      1. Forward Pass                │
│      2. Compute Loss                │
│      3. Backward Pass               │
│      4. Gradient Clipping           │
│      5. Optimizer Step              │
│      6. Scheduler Step              │
│                                     │
│    Every 5 epochs:                  │
│    - Save checkpoint                │
│    - Validate on val_set            │
│    - Log metrics                    │
│                                     │
└────────────┬────────────────────────┘
             │
             ▼
CHECKPOINTING
├─ Checkpoint every 5 epochs
├─ Save: model state + optimizer state
├─ Location: models/checkpoints/
└─ Resume capability

             │
             ▼
FINAL MODEL PACKAGING
├─ config.json (architecture)
├─ model.pt (weights + optimizer)
├─ metadata.json (info)
├─ README.md (usage)
└─ Location: models/final/
```

---

## Summary

### High-Level: Simple 7-Stage Flow
Kannada Text → Dual Encoders → Fusion → Prosody → Posterior → Flow → Vocoder → Audio

### Low-Level: Multiple Components
- **Grapheme Encoder**: 4-layer Transformer
- **Phoneme Encoder**: 2-layer BiLSTM
- **Fusion**: Multi-head Attention
- **Prosody**: Pitch + Energy quantization
- **Posterior**: 4-layer CNN
- **Flow**: 4x invertible transformations
- **Vocoder**: HiFi-GAN with MPD/MSD discriminators

### Data Paths
- Training: Multiple losses computed in parallel
- Inference: Simple deterministic forward pass
- Kannada Processing: Specialized G2P with language rules
