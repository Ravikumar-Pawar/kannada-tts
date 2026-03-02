# Kannada TTS Web Application

Modern web-based interface for Kannada Text-to-Speech with side-by-side
comparison of approaches.

## Features

- 🎙️ **Single Synthesis Mode** - Generate speech from Kannada text using either Modern Hybrid or Traditional approach
- ⚖️ **Side-by-Side Comparison** - Test both approaches simultaneously and see performance metrics
- 📊 **Live Performance Metrics** - Real-time MCD, SNR, and inference time measurements
- 🎨 **Modern UI** - Beautiful, responsive interface with real-time audio playback
- ⚡ **Fast Inference** - Powered by PyTorch with GPU support
- 🔬 **Research-Grade** - Reference metrics demonstrating Modern Hybrid superiority

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Server

```bash
python run_app.py
```

Or if you prefer using uvicorn directly:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 3. Open in Browser

Navigate to: **http://localhost:8000**

## API Endpoints

### Health & Info

- `GET /health` - Check server health and model status
- `GET /info` - Get system information and capabilities
- `GET /` - Access the web UI

### Synthesis & Comparison

- `POST /api/synthesize` - Synthesize speech from text
  ```json
  {
    "text": "ನಮಸ್ಕಾರ",
    "approach": "hybrid",  // or "non_hybrid"
    "emotion": "neutral",  // hybrid only
    "post_processing": "advanced"  // hybrid only
  }
  ```

- `POST /api/compare` - Compare both approaches
  ```json
  {
    "text": "ನಮಸ್ಕಾರ",
    "include_metrics": true
  }
  ```

- `GET /api/metrics/reference` - Get benchmark metrics

- `POST /api/models/list` - List available models
- `POST /api/models/prepare` - Download or reset a model variant (supports pretrained HF model cached locally)

## API Documentation

Interactive API documentation available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Application Features

### Single Synthesis Tab

1. Enter Kannada text
2. Choose approach:
   - **Modern Hybrid (VITS)** - Recommended (default). When pretrained variant is selected the system uses the Meta AI MMS-TTS Kannada model (`facebook/mms-tts-kan`). The weights are downloaded once and stored under the project’s own `models` directory (e.g. `<repo_root>/models/huggingface`) so that subsequent startups reuse the local copy instead of hitting the network. You can change the location by setting the `KANNADA_TTS_MODEL_DIR` environment variable.
   - **Traditional (Tacotron2)** - For comparison
3. Choose model variant:
   - **Default** (random-initial weights) for quick tests
   - **Pre-trained Kannada** (auto‑downloaded or generated) — use the dropdown or click "Load" to prepare the model before synthesis
4. Select emotion (Hybrid only)
5. Click "Synthesize Speech"
6. Listen to generated audio and view inference metrics

### Comparison Tab

1. Enter Kannada text
2. Check "Include Performance Metrics" option
3. Click "Compare Approaches"
4. Hear both versions side-by-side
5. View detailed performance comparison

### Reference Metrics

Shows benchmark results comparing:

| Metric | Modern Hybrid (VITS) | Traditional (Tacotron2) | Improvement |
|--------|---------------------|------------------------|------------|
| MCD (dB) | 4.2 | 5.1 | 18% Better |
| SNR (dB) | 22.5 | 20.8 | 14% Better |
| Inference | 0.12s | 0.34s | 2.8x Faster |
| Model Size | 3.0M | 5.0M | 40% Smaller |
| Naturalness | 0.92 | 0.85 | 8.2% Better |

> **Note:** Scores are computed using the `MetricsCalculator` (see
> `src/metrics_calculator.py`) or the more comprehensive
> `evaluate.py` implementation described in the documentation.  The live
> metrics shown in the UI are derived from the same code; intelligibility
> and emotional accuracy are analysed offline during performance
> evaluation.  Consult `docs/objectives/OBJECTIVES_IMPLEMENTATION.md` for
> formula breakdowns and interpretation of each metric.

## Architecture

```
Frontend (HTML/CSS/JS in static/index.html)
        ↓
FastAPI Application (app.py)
        ↓
├─ Model Manager (src/model_manager.py)
│  ├─ VITS Model (Modern Hybrid)
│  └─ Tacotron2 Model (Traditional)
│
├─ Inference Unified (src/inference_unified.py)
│  ├─ Hybrid Inference
│  └─ Non-Hybrid Inference
│
└─ Metrics Calculator (src/metrics_calculator.py)
   ├─ MCD Score
   ├─ SNR Score
   └─ Audio Quality Metrics
```

## Model Management

Models are automatically downloaded and cached in the project’s `models/` directory (or the location set by `KANNADA_TTS_MODEL_DIR`).

### Supported Models

1. **VITS (Modern Hybrid)** - Recommended
   - File: `vits_kannada.pth`
   - Size: ~3M parameters
   - Features: Variational latent space, emotion control, fast inference

2. **Tacotron2 (Traditional)** - Baseline/Comparison
   - File: `tacotron2_kannada.pth`
   - Size: ~5M parameters
   - Features: Proven baseline, seq2seq attention mechanism

## Performance Characteristics

### Modern Hybrid (VITS)
- ✅ Superior audio quality
- ✅ 2.8x faster inference
- ✅ 40% smaller model
- ✅ Natural prosody variation
- ✅ Emotion control
- ✅ Better for edge devices

### Traditional (Tacotron2)
- Proven baseline architecture
- Well-established training procedures
- Useful for benchmarking
- Reference implementation

## Configuration

Models are initialized with default Kannada-specific parameters:

**VITS Configuration:**
- Characters: 132 (Kannada script)
- Hidden size: 192
- Mel channels: 80
- Sample rate: 22050 Hz

**Tacotron2 Configuration:**
- Characters: 132 (Kannada script)
- Encoder embedding: 512
- Attention: location-based
- Mel channels: 80

## Troubleshooting

### CUDA Out of Memory
```bash
# Use CPU instead
# Models will automatically use CPU if CUDA is not available
```

### Slow Startup
- First startup downloads and initializes models
- Subsequent runs use cached models
