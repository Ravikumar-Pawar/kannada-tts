---
title: "Model Deployment & Integration Guide"
description: "How to use trained HKL-VITS models in other projects"
---

# HKL-VITS Model Deployment & Integration Guide

## Overview

After training completes, your model is saved in a **self-contained package** that can be easily integrated into other projects.

```
models/
└── final/
    └── hkl_vits_20260424_143022/
        ├── config.json                    # Complete configuration
        ├── model.pt                       # Trained weights + optimizer state
        ├── metadata.json                  # Model information
        ├── README.md                      # Usage instructions
        ├── requirements_inference.txt     # Minimal dependencies
        └── architecture_notes.txt         # Architecture details
```

---

## Part 1: Model Package Contents

### 1.1 Configuration File (`config.json`)

Contains all hyperparameters used during training:

```json
{
  "model": {
    "vocab_size": 150,
    "phoneme_vocab_size": 80,
    "hidden_dim": 256,
    "num_mels": 80,
    "sample_rate": 22050,
    "grapheme_encoder": { ... },
    "phoneme_encoder": { ... },
    "fusion_layer": { ... },
    "prosody_encoder": { ... },
    "flow_model": { ... },
    "decoder": { ... }
  },
  "training": { ... },
  "loss_weights": { ... }
}
```

**Use this to**: Recreate the exact model architecture.

### 1.2 Model Weights (`model.pt`)

PyTorch checkpoint containing:
- `model_state_dict`: Trained neural network weights
- `optimizer_state_dict`: Optimizer state (useful for resume)
- `config`: Full training configuration

**Size**: ~350-400 MB (depending on model size)

### 1.3 Metadata (`metadata.json`)

```json
{
  "model_name": "HKL-VITS (Hybrid Linguistic-Enhanced VITS)",
  "language": "Kannada",
  "created": "2026-04-24T14:30:22",
  "version": "1.0",
  "unique_features": [
    "Dual Linguistic Encoders (Grapheme + Phoneme)",
    "Kannada-specific Grapheme-to-Phoneme Conversion",
    "Prosody Conditioning (Pitch + Energy)",
    "Flow-based Latent Variable Model",
    "HiFi-GAN Neural Vocoding"
  ]
}
```

---

## Part 2: Integration into Other Projects

### 2.1 Simple Integration Pattern

**Scenario**: You want to use the trained model in a new Kannada TTS application.

#### Step 1: Copy Model Package

```bash
# Option A: Copy entire package
cp -r /original/project/models/final/hkl_vits_* /new/project/models/

# Option B: Copy only what's needed for inference
mkdir -p /new/project/models/hkl_vits_kannada
cp /original/project/models/final/hkl_vits_*/config.json /new/project/models/
cp /original/project/models/final/hkl_vits_*/model.pt /new/project/models/
```

#### Step 2: Create Simple Wrapper

```python
# inference_wrapper.py
import json
from pathlib import Path
from hkl_vits.inference import HKLVITSInference

class Kannada_TTS:
    """Simple wrapper for trained HKL-VITS model"""
    
    def __init__(self, model_dir='models/hkl_vits_kannada'):
        self.model_dir = Path(model_dir)
        self.inference = HKLVITSInference(
            config_path=self.model_dir / 'config.json',
            checkpoint_path=self.model_dir / 'model.pt',
            device='cuda'
        )
    
    def speak(self, kannada_text: str, output_file: str = None):
        """Generate speech from Kannada text"""
        audio = self.inference.synthesize(kannada_text)
        
        if output_file:
            self.inference.save_audio(audio, output_file)
        
        return audio

# Usage
tts = Kannada_TTS()
audio = tts.speak("ಒಂದು ಮತ್ತು ಎರಡು", "output.wav")
```

#### Step 3: Use in Your Application

```python
from inference_wrapper import Kannada_TTS

# Initialize once
tts = Kannada_TTS()

# Use wherever needed
for text in kannada_texts:
    audio = tts.speak(text)
    process_audio(audio)
```

---

### 2.2 Advanced Integration: Multi-Speaker Support

**Scenario**: You want to adapt the model to new speakers.

```python
# fine_tune_speaker.py
import torch
from pathlib import Path
from hkl_vits.hkl_vits_model import HKLVITS
from hkl_vits.dataset_loader import get_dataloaders
from training.train_hkl_vits import HKLVITSTrainer

class SpeakerAdapter:
    """Adapt pretrained model to new speaker"""
    
    def __init__(self, pretrained_model_dir):
        self.model_dir = Path(pretrained_model_dir)
        self.config = self._load_config()
        self.model = self._load_model()
    
    def _load_config(self):
        import json
        with open(self.model_dir / 'config.json') as f:
            return json.load(f)
    
    def _load_model(self):
        # Build model from config
        from hkl_vits.hkl_vits_model import HKLVITS
        model = HKLVITS(**self.config['model'])
        
        # Load pretrained weights
        checkpoint = torch.load(self.model_dir / 'model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def fine_tune(self, speaker_data_dir, epochs=10):
        """
        Fine-tune on new speaker
        
        Args:
            speaker_data_dir: Path to {speaker_name}/wav and {speaker_name}/txt
            epochs: Number of fine-tuning epochs
        """
        # Create dataloaders for speaker data
        train_loader, val_loader = get_dataloaders(
            speaker_data_dir,
            batch_size=4,  # Small for fine-tuning
            num_workers=0
        )
        
        # Fine-tune
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        
        for epoch in range(epochs):
            for batch in train_loader:
                # Training step
                loss = self._training_step(batch, optimizer)
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}")
        
        return self.model
    
    def save_adapted_model(self, output_dir):
        """Save speaker-adapted model"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save config
        import json
        with open(Path(output_dir) / 'config.json', 'w') as f:
            json.dump(self.config, f)
        
        # Save weights
        torch.save(
            {'model_state_dict': self.model.state_dict()},
            Path(output_dir) / 'model.pt'
        )

# Usage
adapter = SpeakerAdapter('models/hkl_vits_kannada')
new_speaker_model = adapter.fine_tune('data/new_speaker_data', epochs=5)
adapter.save_adapted_model('models/speaker_adapted')
```

---

### 2.3 Production Deployment: REST API

**Scenario**: You want to serve the model as a web service.

```python
# app.py - Flask API
from flask import Flask, request, jsonify
from pathlib import Path
import io
import base64
from hkl_vits.inference import HKLVITSInference

app = Flask(__name__)

# Load model once at startup
inference = HKLVITSInference(
    config_path='models/hkl_vits_kannada/config.json',
    checkpoint_path='models/hkl_vits_kannada/model.pt',
    device='cuda'
)

@app.route('/synthesize', methods=['POST'])
def synthesize():
    """
    Synthesize Kannada text to speech
    
    Request: {
        "text": "ಕನ್ನಡ ಪಠ್ಯ",
        "format": "wav",  # or mp3
        "length_scale": 1.0
    }
    
    Response: {
        "audio": "base64_encoded_audio",
        "duration": 2.5,
        "format": "wav"
    }
    """
    data = request.json
    kannada_text = data.get('text', '')
    
    if not kannada_text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Generate speech
    audio = inference.synthesize(
        kannada_text,
        length_scale=data.get('length_scale', 1.0)
    )
    
    # Encode to base64
    audio_bytes = io.BytesIO()
    import torchaudio
    torchaudio.save(audio_bytes, audio, inference.config['model']['sample_rate'], format='wav')
    audio_b64 = base64.b64encode(audio_bytes.getvalue()).decode()
    
    return jsonify({
        'audio': audio_b64,
        'duration': audio.shape[-1] / inference.config['model']['sample_rate'],
        'format': 'wav'
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': 'HKL-VITS', 'language': 'Kannada'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

**Usage**:
```python
import requests
import json
import base64

response = requests.post('http://localhost:5000/synthesize', json={
    'text': 'ನಮಸ್ಕಾರ ಜನರೇ',
    'length_scale': 1.0
})

# Decode and save audio
if response.status_code == 200:
    audio_data = base64.b64decode(response.json()['audio'])
    with open('output.wav', 'wb') as f:
        f.write(audio_data)
```

---

### 2.4 Batch Processing Pipeline

**Scenario**: You need to synthesize many Kannada texts.

```python
# batch_synthesize.py
import torch
from pathlib import Path
from tqdm import tqdm
from hkl_vits.inference import HKLVITSInference

class BatchSynthesizer:
    def __init__(self, model_dir):
        self.inference = HKLVITSInference(
            config_path=f'{model_dir}/config.json',
            checkpoint_path=f'{model_dir}/model.pt',
            device='cuda'
        )
    
    def process_file(self, input_file, output_dir):
        """
        Process file with Kannada texts (one per line)
        Generate audio for each line
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for i, kannada_text in enumerate(tqdm(lines, desc='Synthesizing')):
            kannada_text = kannada_text.strip()
            
            if not kannada_text:
                continue
            
            try:
                audio = self.inference.synthesize(kannada_text)
                
                output_path = output_dir / f'audio_{i:04d}.wav'
                self.inference.save_audio(audio, str(output_path))
                
            except Exception as e:
                print(f"Error processing line {i}: {e}")
    
    def process_directory(self, input_dir, output_dir):
        """Process all text files in directory"""
        input_dir = Path(input_dir)
        text_files = sorted(input_dir.glob('*.txt'))
        
        for text_file in tqdm(text_files, desc='Processing files'):
            output_subdir = Path(output_dir) / text_file.stem
            self.process_file(text_file, output_subdir)

# Usage
synthesizer = BatchSynthesizer('models/hkl_vits_kannada')
synthesizer.process_file('kannada_texts.txt', 'output_audios')
```

---

## Part 3: Model Optimization for Deployment

### 3.1 Quantization (Reduce Model Size)

```python
# quantize_model.py
import torch
from pathlib import Path

def quantize_model(model_dir, output_dir):
    """Quantize model to reduce size and improve speed"""
    
    # Load model
    from hkl_vits.hkl_vits_model import HKLVITS
    import json
    
    with open(f'{model_dir}/config.json') as f:
        config = json.load(f)
    
    model = HKLVITS(**config['model'])
    checkpoint = torch.load(f'{model_dir}/model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Quantize to 8-bit
    model_q = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    
    # Save quantized model
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model_q.state_dict(), f'{output_dir}/model_quantized.pt')
    
    # Calculate size reduction
    original_size = Path(f'{model_dir}/model.pt').stat().st_size / 1e6
    quantized_size = Path(f'{output_dir}/model_quantized.pt').stat().st_size / 1e6
    
    print(f"Original size: {original_size:.1f} MB")
    print(f"Quantized size: {quantized_size:.1f} MB")
    print(f"Reduction: {(1 - quantized_size/original_size)*100:.1f}%")

# Usage
quantize_model('models/hkl_vits_kannada', 'models/hkl_vits_kannada_quantized')
```

### 3.2 Export to ONNX (Cross-Platform)

```python
# export_onnx.py
import torch
import json

def export_to_onnx(model_dir, output_path):
    """Export model to ONNX format for deployment"""
    
    from hkl_vits.hkl_vits_model import HKLVITS
    
    with open(f'{model_dir}/config.json') as f:
        config = json.load(f)
    
    # Build model
    model = HKLVITS(**config['model'])
    checkpoint = torch.load(f'{model_dir}/model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy inputs
    batch_size = 1
    seq_len = 100
    text = torch.randint(0, config['model']['vocab_size'], (batch_size, seq_len))
    phonemes = torch.randint(0, config['model']['phoneme_vocab_size'], (batch_size, seq_len))
    
    # Export to ONNX
    torch.onnx.export(
        model,
        (text, phonemes),
        output_path,
        input_names=['text', 'phonemes'],
        output_names=['mel_output', 'alignments'],
        opset_version=12
    )
    
    print(f"Model exported to {output_path}")

# Usage
export_to_onnx('models/hkl_vits_kannada', 'models/hkl_vits_kannada.onnx')
```

---

## Part 4: Evaluation & Validation

### 4.1 Audio Quality Assessment

```python
# evaluate_model.py
import torch
import librosa
import numpy as np
from scipy.spatial.distance import euclidean

def evaluate_synthesis(generated_audio, reference_audio, sr=22050):
    """
    Compare generated audio with reference
    Returns quality metrics
    """
    
    # Mel-scaled spectrograms
    mel_gen = librosa.feature.melspectrogram(y=generated_audio, sr=sr)
    mel_ref = librosa.feature.melspectrogram(y=reference_audio, sr=sr)
    
    # Log scale
    mel_gen_db = librosa.power_to_db(mel_gen, ref=np.max)
    mel_ref_db = librosa.power_to_db(mel_ref, ref=np.max)
    
    # Interpolate to same length
    min_len = min(mel_gen_db.shape[1], mel_ref_db.shape[1])
    mel_gen_db = mel_gen_db[:, :min_len]
    mel_ref_db = mel_ref_db[:, :min_len]
    
    # Metrics
    mse = np.mean((mel_gen_db - mel_ref_db) ** 2)
    l1 = np.mean(np.abs(mel_gen_db - mel_ref_db))
    
    return {
        'mel_mse': mse,
        'mel_l1': l1,
        'duration': len(generated_audio) / sr
    }

# Usage
from hkl_vits.inference import HKLVITSInference

inference = HKLVITSInference('config.json', 'model.pt')

# Test synthesis
text = "ಕನ್ನಡ"
audio_generated = inference.synthesize(text)

# Load reference (if available)
reference_audio, sr = librosa.load('reference.wav')

# Evaluate
metrics = evaluate_synthesis(audio_generated.numpy(), reference_audio)
print(f"Metrics: {metrics}")
```

---

## Part 5: Troubleshooting

### Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: hkl_vits` | Package not in Python path | Add project to PYTHONPATH or sys.path |
| `RuntimeError: CUDA out of memory` | Model too large for GPU | Use CPU, reduce batch size, or quantize |
| `OSError: model.pt not found` | Wrong path | Verify model_dir path is correct |
| `Poor audio quality` | Incomplete training | Train longer or check data quality |
| `Slow inference` | CPU inference | Use CUDA device or optimize model |

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from hkl_vits.inference import HKLVITSInference

inference = HKLVITSInference(
    'config.json',
    'model.pt',
    device='cuda'
)

# Will print detailed debug info
audio = inference.synthesize("ಟೆಸ್ಟ್")
```

---

## Summary Checklist

- ✅ Model saved in `models/final/`
- ✅ Copy model to new project or use path reference
- ✅ Load with `HKLVITSInference` class
- ✅ Test on sample Kannada text
- ✅ Evaluate audio quality
- ✅ Consider optimization (quantization, ONNX)
- ✅ Deploy in production or share with others
- ✅ Document any modifications or integrations

---

## Further Reading

- [Inference Module Documentation](../hkl_vits/inference.py)
- [Model Architecture](../hkl_vits/hkl_vits_model.py)
- [Unique Features Research](./UNIQUE_FEATURES_RESEARCH.md)
- [Project Summary](./PROJECT_SUMMARY.md)

---

**Last Updated**: 2026-04-24  
**Status**: Production Ready
