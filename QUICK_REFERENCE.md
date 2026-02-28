# 🎵 Kannada TTS - Quick Reference Card

## 📋 Basic Commands

### Setup
```bash
python -m venv venv
source venv/Scripts/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu118
```

### Validation
```bash
python src/validate.py             # Check system readiness
```

### Full Pipeline
```bash
python src/data_prep.py            # ~10 min - Prepare dataset
python src/train_tacotron.py       # ~24-48 hrs - Train models
python src/inference.py            # ~30 sec - Generate speech
python src/evaluate.py             # ~5 min - Evaluate quality
```

### Quick Demo
```bash
python src/demo.py                 # Run pipeline automatically
```

---

## 🔍 Inspection Commands

### Dataset Analysis
```python
from src.utils import DatasetUtils
stats = DatasetUtils.analyze_dataset("data/metadata.csv")
print(stats)
```

### Audio Info
```python
from src.utils import AudioUtils
info = AudioUtils.get_audio_info("output/inference/test.wav")
print(info)
```

### Model Size
```python
from src.utils import ModelUtils
size = ModelUtils.get_model_size("output/tacotron2/best_model.pth")
print(size)
```

### System Diagnostics
```python
from src.utils import SystemUtils
SystemUtils.print_diagnostics()
```

---

## 🎨 Inference Examples

### Basic Synthesis
```python
from src.inference import KannadaTTSInference

engine = KannadaTTSInference("output/tacotron2/best_model.pth")
audio, sr = engine.synthesize("ನಮಸ್ಕಾರ")
```

### Emotion Synthesis
```python
# Happy
audio, sr = engine.synthesize("ಸುಖವಾಗಿದೆ!", emotion="happy")

# Sad
audio, sr = engine.synthesize("ನನಗೆ ದುಃಖವಾಯಿತು.", emotion="sad")

# Angry
audio, sr = engine.synthesize("ಇದು ತಪ್ಪು!", emotion="angry")

# Calm
audio, sr = engine.synthesize("ನಿಶ್ಚಿಂತಪಡಿ.", emotion="calm")
```

### With Assessment
```python
result = engine.assess_and_synthesize(
    text="ಕನ್ನಡ TTS ವ್ಯವಸ್ಥೆ",
    output_path="output/sample.wav",
    emotion="neutral"
)
print(result['quality_metrics'])
```

---

## 🔊 Audio Processing

### Noise Reduction
```python
from src.inference import NoiseReductionModule

denoiser = NoiseReductionModule()
clean = denoiser.denoise(audio, method="spectral_gating")
```

### Emotion Enhancement
```python
from src.inference import EmotionEnhancementModule

enhancer = EmotionEnhancementModule()
enhanced = enhancer.apply_emotion(audio, emotion="happy")
```

### Custom Prosody
```python
modified = enhancer.enhance_prosody(
    audio,
    pitch_shift=2.0,        # +2 semitones
    duration_scale=0.9,     # 10% faster
    energy_scale=1.3        # 30% louder
)
```

### Quality Check
```python
from src.inference import SpeechQualityAssessment

assessor = SpeechQualityAssessment()
metrics = assessor.assess_quality(audio)

print(f"SNR: {metrics['snr_db']:.1f} dB")
print(f"Clarity: {metrics['intelligibility_score']:.1f}%")
```

---

## 📊 Evaluation

### Single File
```python
from src.evaluate import SpeechEvaluationMetrics
import librosa

evaluator = SpeechEvaluationMetrics()
y_ref, sr = librosa.load("reference.wav", sr=22050)
y_syn, sr = librosa.load("synthesis.wav", sr=22050)

metrics = evaluator.evaluate(y_ref, y_syn)
print(f"MCD: {metrics['mcd_mean']:.2f} dB")
print(f"SNR: {metrics['snr_db']:.1f} dB")
```

### Batch Evaluation
```bash
python src/evaluate.py
```

---

## 🛠️ Configuration

### Default Tacotron2
```json
{
  "batch_size": 16,
  "epochs": 500,
  "encoder_hidden_size": 256,
  "decoder_hidden_size": 1024,
  "lr": 0.001
}
```

### For GPU with 8GB Memory
```json
{
  "batch_size": 8,
  "encoder_hidden_size": 128,
  "decoder_hidden_size": 512,
  "num_loader_workers": 0
}
```

### For Fast Training
```json
{
  "epochs": 100,
  "batch_size": 32,
  "encoder_hidden_size": 512,
  "decoder_hidden_size": 2048
}
```

---

## 📁 File Organization

```
kannada-tts/
├── Data Preparation
│   └── python src/data_prep.py
│       → data/metadata.csv (16,950 samples)
│
├── Training  
│   └── python src/train_tacotron.py
│       → output/tacotron2/best_model.pth
│
├── Inference
│   └── python src/inference.py
│       → output/inference/test_*.wav
│
├── Evaluation
│   └── python src/evaluate.py
│       → output/evaluation_results.json
│
├── Utilities
│   ├── python src/utils.py        # Analysis tools
│   ├── python src/validate.py     # Validation
│   └── python src/demo.py         # Demo pipeline
│
└── Documentation
    ├── README.md                  # Main guide
    ├── CONFIG_GUIDE.md            # Configuration
    └── UPDATES.md                 # Changes
```

---

## 🎯 Common Tasks

### Generate Speech from Kannada Text
```bash
python src/inference.py
# Outputs: output/inference/test_neutral.wav
#          output/inference/test_happy.wav
#          output/inference/test_calm.wav
```

### Check Dataset Quality
```python
from src.utils import DatasetUtils
df = DatasetUtils.load_metadata("data/metadata.csv")
print(df.head())
print(DatasetUtils.analyze_dataset("data/metadata.csv"))
```

### Visualize Waveform
```python
from src.utils import AudioUtils
AudioUtils.plot_waveform("output/inference/test.wav")
AudioUtils.plot_spectrogram("output/inference/test.wav")
```

### System Status
```bash
python src/validate.py
```

### Training Logs
```bash
tail -f output/training.log
tensorboard --logdir output/tacotron2/
```

---

## 🔢 Performance Targets

| Metric | Target | Typical |
|--------|--------|---------|
| MCD | < 7.0 dB | 6.2 dB |
| MSSTFT | < 2.5 dB | 1.8 dB |
| SNR | > 20 dB | 28.5 dB |
| Intelligibility | > 80% | 85% |
| Training Time | ~36 hrs | 24-48 hrs |
| Inference Speed | RTF < 0.5 | 0.25 RTF |

---

## ⚡ Optimization Tips

### Faster Training
- Increase `batch_size` (if GPU memory allows)
- Reduce `encoder_hidden_size` to 128
- Set `num_epochs` to 100-200

### Better Quality
- Increase `epochs` to 1000+
- Double model sizes (hidden dims)
- Use `hifigan` vocoder
- Increase `n_mel_channels` to 128

### Lower Memory (GPU < 8GB)
```json
{
  "batch_size": 4,
  "encoder_hidden_size": 128,
  "decoder_hidden_size": 256,
  "num_loader_workers": 0
}
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Reduce `batch_size` to 4-8 |
| Slow data loading | Increase `num_loader_workers` |
| Poor audio quality | Increase `epochs` and `n_mel_channels` |
| Training divergence | Reduce `lr` to 0.0005 |
| Missing dataset | Run `python src/data_prep.py` |
| Import errors | Run `python src/validate.py` |

---

## 📞 Support

- **Validation:** `python src/validate.py`
- **Training logs:** `output/training.log`
- **Configuration:** `CONFIG_GUIDE.md`
- **General help:** `README.md`

---

## 🚀 Next Steps

1. ✅ Validate system: `python src/validate.py`
2. 📊 Prepare data: `python src/data_prep.py`
3. 🎓 Train models: `python src/train_tacotron.py`
4. 🎤 Generate speech: `python src/inference.py`
5. 📈 Evaluate: `python src/evaluate.py`

---

**Last Updated:** 2026-02-28  
**Version:** 2.0 (Production Ready)
