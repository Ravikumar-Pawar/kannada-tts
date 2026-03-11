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

## Project Structure

```
kannada-hkl-vits/
├── project_guide.txt              # Comprehensive technical guide
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
│   └── evaluate.py                # Evaluation metrics
├── data/                          # Dataset directory (create this)
│   ├── wav/                       # Audio files
│   └── txt/                       # Text transcriptions
└── logs/                          # Training logs and checkpoints
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- torchaudio
- librosa
- numpy
- scipy

### Setup

```bash
# Clone repository
git clone <repo_url>
cd kannada-hkl-vits

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchaudio librosa numpy scipy
```

## Dataset Preparation

### Expected Dataset Structure

```
dataset/
├── wav/                    # Audio files (.wav format)
│   ├── sample_001.wav
│   ├── sample_002.wav
│   └── ...
└── txt/                    # Text transcriptions
    ├── sample_001.txt
    ├── sample_002.txt
    └── ...
```

### Audio Requirements

- **Sample Rate**: 22050 Hz (configurable)
- **Format**: WAV (mono or stereo)
- **Duration**: 1-10 seconds per sample recommended
- **Quality**: High quality, minimal background noise

### Text Requirements

- **Encoding**: UTF-8
- **Language**: Kannada script
- **Format**: Plain text (one line per file)

### Create Dataset

```bash
python hkl_vits/utils.py prepare_dataset --data_dir path/to/dataset
```

## Training

### Basic Training

```bash
python training/train_hkl_vits.py \
    --config configs/hkl_vits_config.json \
    --data_dir path/to/dataset \
    --num_epochs 100 \
    --gpu 0
```

### Training Configuration

Edit `configs/hkl_vits_config.json` to customize:

```json
{
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

### Resume Training

```bash
python training/train_hkl_vits.py \
    --config configs/hkl_vits_config.json \
    --data_dir path/to/dataset \
    --resume checkpoints/hkl_vits_epoch_50.pt \
    --gpu 0
```

## Inference

### Single Text Synthesis

```bash
python hkl_vits/inference.py \
    --config configs/hkl_vits_config.json \
    --checkpoint checkpoints/hkl_vits_epoch_100.pt \
    --text "ನಮಸ್ತೆ ಪ್ರಪಂಚ" \
    --output output.wav
```

### Interactive Synthesis

```bash
python hkl_vits/inference.py \
    --config configs/hkl_vits_config.json \
    --checkpoint checkpoints/hkl_vits_epoch_100.pt \
    --interactive
```

### Batch Synthesis

```python
from hkl_vits.inference import HKLVITSInference

inference = HKLVITSInference(
    config_path='configs/hkl_vits_config.json',
    checkpoint_path='checkpoints/hkl_vits_epoch_100.pt'
)

texts = [
    'ನಮಸ್ತೆ',
    'ಧನ್ಯವಾದ',
    'ಎಲ್ಲಾ ಪ್ರಶ್ನೆಗಳಿಗೆ'
]

waveforms = inference.synthesize_batch(texts, save_dir='outputs')
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
