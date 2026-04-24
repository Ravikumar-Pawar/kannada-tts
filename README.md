# HKL-VITS: Hybrid Linguistic-Enhanced VITS for Kannada Text-to-Speech

A state-of-the-art Text-to-Speech (TTS) system for Kannada that combines multiple linguistic representations (grapheme, phoneme, and prosody) with advanced neural vocoding techniques.

## Project Overview

HKL-VITS (Hybrid Linguistic-Enhanced VITS) is built on the VITS architecture but extends it with:

- **Dual Linguistic Encoders**: Separate grapheme and phoneme encoders that capture both spelling structure and pronunciation
- **Fusion Layer**: Intelligently combines grapheme and phoneme representations
- **Prosody Conditioning**: Models pitch (F0) and energy for natural prosody
- **End-to-End Training**: Single unified loss function combining reconstruction, KL divergence, adversarial, pitch, and energy losses

## Architecture

```
Kannada Text Input
    ↓
Grapheme Encoder (Transformer)  +  Phoneme Encoder (BiLSTM)
    ↓
Fusion Layer (Linear + Attention)
    ↓
+ Prosody Encoder (Pitch + Energy)
    ↓
Posterior Encoder (Mel-spectrogram)
    ↓
Flow-Based Model
    ↓
HiFi-GAN Generator
    ↓
Waveform Output
```

## Quick Start

Get started in minutes:

```bash
# 1. Setup (5 min)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Download Dataset (10-30 min)
python dataset.py full

# 3. Configure (1 min)
# Edit configs/hkl_vits_config.json and set:
# "data.dataset_path": "data/kannada_tts_dataset"

# 4. Train (hours to days - just run it!)
python training/train_hkl_vits.py

# 5. Evaluate (auto-finds latest checkpoint)
python training/evaluate.py

# 6. Generate Speech (interactive mode)
python hkl_vits/inference.py
```

**That's it!** No complex arguments needed. Everything is configured in `configs/hkl_vits_config.json`.

## Project Structure

```
kannada-hkl-vits/
├── project_guide.txt              # Comprehensive technical guide
├── dataset.py                     # Dataset management tool
├── requirements.txt               # Python dependencies
├── configs/
│   └── hkl_vits_config.json       # Model and training configuration
├── hkl_vits/
│   ├── grapheme_encoder.py        # Grapheme to embedding encoder
│   ├── phoneme_encoder.py         # Phoneme to embedding encoder
│   ├── fusion_layer.py            # Fusion of linguistic representations
│   ├── prosody_encoder.py         # Pitch and energy conditioning
│   ├── hkl_vits_model.py          # Main VITS model
│   ├── kannada_g2p.py             # Grapheme-to-Phoneme conversion
│   ├── dataset_loader.py          # Data loading and preprocessing
│   ├── loss_functions.py          # Loss computation
│   ├── inference.py               # Inference and synthesis
│   └── utils.py                   # Utility functions
├── training/
│   ├── train_hkl_vits.py          # Training script
│   └── evaluate.py                # Evaluation and testing metrics
├── data/                          # Dataset directory
│   └── kannada_tts_dataset/
│       ├── wav/                   # Audio files
│       ├── txt/                   # Text transcriptions
│       └── metadata.tsv           # Dataset metadata
├── logs/                          # Training logs
├── checkpoints/                   # Model checkpoints
└── docs/                          # Documentation
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- GPU with CUDA support (optional but recommended)

### Step-by-Step Setup

```bash
# 1. Clone repository and navigate to directory
cd kannada-hkl-vits

# 2. Create virtual environment
python -m venv venv

# Activate (choose based on OS):
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch; print('✓ PyTorch:', torch.__version__); print('✓ CUDA Available:', torch.cuda.is_available())"
```

## Dataset Download and Preparation

### Option 1: Automated Download from OpenSLR (Recommended)

The easiest way to get the Kannada TTS dataset from [OpenSLR](https://openslr.org/):

```bash
# Download + Extract + Prepare (all in one, ~20GB download)
python dataset.py full

# Or step by step:

# Step 1: Download dataset (~20GB)
python dataset.py download

# Step 2: Extract and prepare text files
python dataset.py prepare

# Step 3: Analyze dataset (optional, generates statistics)
python dataset.py analyze --data-dir data/kannada_tts_dataset --save-report
```

**Dataset Details:**
- **Source**: OpenSLR - Male Kannada speaker
- **Size**: ~20GB compressed, ~40GB uncompressed
- **Samples**: 1000+ audio files with transcriptions
- **Sample Rate**: 16kHz (will be resampled to 22050Hz)
- **Format**: WAV + TSV metadata

### Option 2: Organize Custom Dataset

If you have your own Kannada audio and text data:

```bash
# Organize your data into the correct structure
python dataset.py organize \
    --data-dir data/kannada_tts_dataset \
    --audio-source /path/to/your/audio/files \
    --text-source /path/to/your/text/files \
    --sample-rate 22050
```

**Directory Structure Required:**
```
your_dataset/
├── audio_files/
│   ├── sample_001.wav
│   ├── sample_002.wav
│   └── ...
└── text_files/
    ├── sample_001.txt    # Must match audio filename
    ├── sample_002.txt
    └── ...
```

**Text File Requirements:**
- UTF-8 encoding
- Kannada script (Unicode range: U+0C80-U+0CFF)
- One transcription per file
- Plain text format

**Audio File Requirements:**
- WAV format (mono or stereo)
- Recommended sample rate: 22050 Hz (will be resampled if different)
- Duration: 1-10 seconds per sample
- Quality: Clear speech, minimal background noise
- Silence at beginning/end will be trimmed

### Validate Dataset

After preparation, validate your dataset:

```bash
# Analyze and validate
python dataset.py analyze --data-dir data/kannada_tts_dataset --save-report

# This generates statistics:
# - Total samples
# - Audio duration range
# - Text length statistics
# - Missing files
# - Quality issues
```

## Training

All training settings are configured in [configs/hkl_vits_config.json](configs/hkl_vits_config.json). Simply run:

```bash
python training/train_hkl_vits.py
```

That's it! The script will:
- Read all settings from the config file
- Auto-detect GPU or use CPU
- Create logs and checkpoint directories
- Save checkpoints during training

### Configuration

Before training, update the config file with your settings:

```json
{
  "data": {
    "dataset_path": "data/kannada_tts_dataset"  // Update this path to your dataset
  },
  "training": {
    "batch_size": 32,
    "learning_rate": 0.0002,
    "num_epochs": 100,
    "gradient_clip": 1.0
  },
  "loss_weights": {
    "reconstruction": 1.0,
    "kl_divergence": 0.1,
    "adversarial": 1.0,
    "f0": 0.5,
    "energy": 0.1
  }
}
```

### Expected Output

```
════════════════════════════════════════════════════════════
Training Configuration:
════════════════════════════════════════════════════════════
Config: configs/hkl_vits_config.json
Dataset: data/kannada_tts_dataset
Epochs: 100
Batch Size: 32
Learning Rate: 0.0002
Device: cuda:0
════════════════════════════════════════════════════════════

Training logs saved to: logs/training_20240311_143022.log
Checkpoints saved to: checkpoints/

Epoch [1/100]
  Training...
  Saving checkpoint: checkpoints/hkl_vits_epoch_5.pt
  ...
```

## Testing and Evaluation

### Evaluate Model

Simply run the evaluation script - it automatically finds the latest checkpoint:

```bash
python training/evaluate.py
```

Done! The script will:
- Auto-detect the latest checkpoint from `checkpoints/`
- Load the dataset from config
- Run complete evaluation
- Save report to `checkpoints/evaluation_report.json`

### Evaluation Output

```
════════════════════════════════════════════════════════════
Evaluation Configuration:
════════════════════════════════════════════════════════════
Config: configs/hkl_vits_config.json
Checkpoint: checkpoints/hkl_vits_epoch_100.pt
Dataset: data/kannada_tts_dataset
Batch Size: 32
Device: cuda
════════════════════════════════════════════════════════════

✓ Evaluation complete! Report saved to: checkpoints/evaluation_report.json
```

### View Evaluation Results

```bash
# Display the evaluation report
cat checkpoints/evaluation_report.json
```

The report includes:
- **PESQ** (Perceptual Evaluation of Speech Quality)
- **MCD** (Mel-Cepstral Distortion)
- **Pitch Accuracy** metrics
- **Energy Accuracy** metrics
- **Intelligibility** scores

### Test with Sample Sentences

```python
from hkl_vits.inference import HKLVITSInference
from pathlib import Path
import os

# Initialize inference (auto-loads latest checkpoint)
inference = HKLVITSInference(
    config_path='configs/hkl_vits_config.json',
    checkpoint_path='checkpoints/hkl_vits_epoch_100.pt',
    device='cuda'
)

# Test sentences
test_sentences = [
    'ನಮಸ್ತೆ ಪ್ರಪಂಚ',           # Hello World
    'ಧನ್ಯವಾದ',                    # Thank you
    'ಈ ಮಾದರಿಯ ಪ್ರಯೋಗ ಸಾಫಲ್ಯ',  # This experiment is successful
]

# Create output directory
os.makedirs('test_outputs', exist_ok=True)

# Synthesize each test sentence
for i, text in enumerate(test_sentences, 1):
    waveform = inference.synthesize(
        kannada_text=text,
        output_path=f'test_outputs/test_{i:02d}.wav'
    )
    print(f"✓ Generated: test_{i:02d}.wav ({text})")
```
```

## Inference and Synthesis

### Interactive Mode (Default)

Simply run the script with no arguments to start interactive synthesis:

```bash
python hkl_vits/inference.py
```

The script will:
- Auto-load the latest checkpoint
- Auto-detect GPU
- Start interactive mode where you can type Kannada text

Example:
```
Starting Interactive Synthesis Mode...
Type Kannada text and press Enter to synthesize
Type 'exit' to quit

ನಮಸ್ತೆ ಪ್ರಪಂಚ
Synthesizing: ನಮಸ್ತೆ ಪ್ರಪಂಚ
✓ Synthesis complete! Saved to outputs/output_001.wav
```

### Single Sentence Synthesis

Generate speech from a single Kannada sentence:

```bash
# Simple - saves as output.wav
python hkl_vits/inference.py "ನಮಸ್ತೆ ಪ್ರಪಂಚ"

# With custom output filename
python hkl_vits/inference.py "ಧನ್ಯವಾದ" my_output.wav
```

### Batch Synthesis (Python)

Generate speech for multiple texts programmatically:

```python
from hkl_vits.inference import HKLVITSInference
import os

# Initialize (auto-loads latest checkpoint)
inference = HKLVITSInference(
    config_path='configs/hkl_vits_config.json',
    checkpoint_path='checkpoints/hkl_vits_epoch_100.pt',
    device='cuda'  # auto-detected
)

# Synthesize multiple texts
texts = [
    'ನಮಸ್ತೆ',
    'ಧನ್ಯವಾದ',
    'ಎಲ್ಲಾ ಪ್ರಶ್ನೆಗಳಿಗೆ'
]

os.makedirs('outputs', exist_ok=True)
for i, text in enumerate(texts, 1):
    waveform = inference.synthesize(
        kannada_text=text,
        output_path=f'outputs/sample_{i:03d}.wav'
    )
    print(f"✓ {i}. {text}")
```

### Advanced Control

```python
from hkl_vits.inference import HKLVITSInference

inference = HKLVITSInference(
    config_path='configs/hkl_vits_config.json',
    checkpoint_path='checkpoints/hkl_vits_epoch_100.pt'
)

# Synthesize with temperature control (0.5-1.0)
waveform = inference.synthesize(
    kannada_text="ನಮಸ್ತೆ",
    temperature=0.8,  # Lower = more stable
    output_path='output.wav'
)

# Synthesize with custom duration
waveform = inference.synthesize(
    kannada_text="ನಮಸ್ತೆ",
    length_scale=1.2,  # 1.2x slower
    output_path='output_slow.wav'
)
```

## Model Components

### Grapheme Encoder

Transforms Kannada graphemes (characters) into embeddings using:
- Embedding layer (character → vector)
- Positional encoding
- Multi-head Transformer layers

**Output**: $H_g \in \mathbb{R}^{n \times d}$

### Phoneme Encoder

Converts Kannada phonemes into embeddings using:
- Embedding layer
- Bidirectional LSTM layers
- Layer normalization

**Output**: $H_p \in \mathbb{R}^{m \times d}$

### Fusion Layer

Combines grapheme and phoneme representations:

$$H = W[H_g \Vert H_p] + b$$

Options:
- **Linear**: Simple concatenation + linear projection
- **Gated**: Learned gate: $\alpha H_g + (1-\alpha) H_p$
- **Attention**: Cross-attention between phonemes and graphemes

### Prosody Encoder

Models pitch (F0) and energy:
- Continuous or discrete embeddings
- Conv1d processing
- Fusion and normalization

## Loss Functions

### Total Loss

$$L_{total} = \alpha_{recon} L_{recon} + \alpha_{kl} L_{kl} + \alpha_{adv} L_{adv} + \alpha_{f0} L_{f0} + \alpha_{e} L_{e}$$

Where:
- **Reconstruction Loss**: L1 distance between predicted and ground-truth mel-spectrograms
- **KL Divergence**: Regularization of latent space (Gaussian)
- **Adversarial Loss**: LSGAN or standard GAN loss
- **Pitch Loss**: F0 contour accuracy
- **Energy Loss**: Energy contour accuracy

## Kannada Phoneme Inventory

### Vowels (ಸ್ವರಗಳು)
- Short: a, i, u, e, o
- Long: aa, ii, uu, ee, oo
- Diphthongs: ai, au

### Consonants (ವ್ಯಂಜನಗಳು)
- Velar: ka, kha, ga, gha
- Palatal: cha, cha, ja, jha
- Retroflex: tta, ttha, da, dha
- Dental: ta, tha, da, dha
- Labial: pa, pha, ba, bha
- Nasals: na, na, ma
- Approximants: ya, ra, la, va
- Fricatives: sha, ssa, sa, ha

## Kannada-Specific Challenges Addressed

1. **Morphological Complexity**: Grapheme encoder captures compound structures
2. **Vowel Length Contrast**: Phoneme encoder distinguishes short/long vowels
3. **Gemination**: Both encoders handle doubled consonants
4. **Agglutination**: Grapheme structure reveals morphological boundaries

## Evaluation

### Quantitative Metrics

```bash
python training/evaluate.py \
    --checkpoint checkpoints/hkl_vits_epoch_100.pt \
    --test_data path/to/test/set
```

Computes:
- PESQ (Perceptual Evaluation of Speech Quality)
- MCD (Mel-Cepstral Distortion)
- Intelligibility metrics
- Prosody correlation

### Qualitative Evaluation

- Listen to synthesized samples
- Evaluate pronunciation accuracy
- Assess naturalness and prosody
- Check phoneme boundaries

## Advanced Usage

### Custom Prosody

```python
import torch

pitch = torch.tensor([[100, 120, 110, 90]])  # Hz
energy = torch.tensor([[0.5, 0.6, 0.5, 0.4]])

waveform = inference.synthesize(
    kannada_text="ನಮಸ್ತೆ",
    pitch=pitch,
    energy=energy
)
```

### Model Fine-tuning

```python
model = HKLVITS.load('checkpoints/hkl_vits_epoch_100.pt')

# Freeze encoders
for param in model.grapheme_encoder.parameters():
    param.requires_grad = False

# Train only fusion and prosody
optimizer = torch.optim.Adam([
    {'params': model.fusion.parameters()},
    {'params': model.prosody.parameters()}
], lr=1e-5)
```

## Performance

Target Performance Metrics:

- **MOS (Mean Opinion Score)**: > 4.0/5.0
- **Naturalness**: > 85% confidence
- **Intelligibility**: > 95% word accuracy
- **Pitch RMSE**: < 5% of fundamental frequency

## References

### Key Papers

1. Glow-TTS: A Generative Flow for Text-to-Speech based on Generative Flow for Natural Language (Movalglava et al., ICML 2021)
2. VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech (Kim et al., ICML 2021)
3. HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis (Kong et al., NeurIPS 2020)

### Kannada Language Resources

- Kannada Script: IS 13194 (Unicode 0C80-0CFF)
- Kannada Phonemics: Unicode Kannada Block
- Kannada Morphology: Agglutinative structure with suffixes

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## License

[Specify your license]

## Citation

If you use HKL-VITS in your research, please cite:

```bibtex
@software{hklvits2024,
  title={HKL-VITS: Hybrid Linguistic-Enhanced VITS for Kannada Text-to-Speech},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/kannada-hkl-vits}
}
```

## Contact & Support

- Issues: Create GitHub issues for bug reports
- Discussions: Use GitHub discussions for feature requests
- Email: [your-email@example.com]

## Troubleshooting

### Out of Memory (OOM)

- Reduce `batch_size` in config
- Reduce `max_audio_length`
- Use gradient accumulation
- Move to larger GPU

### Poor Quality Audio

- Check data preparation (noise, artifacts)
- Increase training epochs
- Verify loss convergence
- Tune loss weights

### Slow Inference

- Use GPU (set `device='cuda'`)
- Reduce sequence length
- Use FP16 precision (mixed_precision: true)

## Changelog

### v1.0.0 (2024-03-11)

- Initial release
- Grapheme and Phoneme encoders
- Prosody conditioning
- Training and inference pipelines
- Interactive synthesis mode
