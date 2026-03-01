# Integrating Your Own Trained Models

Guide for replacing the default models with your custom trained Kannada TTS models.

## Overview

The application currently initializes models with default configurations. To integrate your own pre-trained or custom-trained models, follow this guide.

## Quick Start

### Option 1: Simple Model Replacement (Recommended)

1. **Train or obtain your models:**
   - VITS model: `your_vits_model.pth`
   - Tacotron2 model: `your_tacotron2_model.pth`

2. **Place models in cache directory:**
   ```bash
   mkdir -p ~/.cache/kannada_tts
   cp your_vits_model.pth ~/.cache/kannada_tts/vits_kannada.pth
   cp your_tacotron2_model.pth ~/.cache/kannada_tts/tacotron2_kannada.pth
   ```

3. **Restart the application:**
   ```bash
   python run_app.py
   ```

The application will automatically load your models on startup.

---

## Detailed Integration Steps

### Step 1: Train Your Models

Use the training interface to train models:

```python
from src.training_unified import TTSTrainer
from src.hybrid.models import VITS

# Train VITS hybrid model
vits = VITS(num_chars=132, hidden_size=192, mel_channels=80)
trainer = TTSTrainer(approach="hybrid", model_type="vits")

# Train for multiple epochs
for epoch in range(100):
    metrics = trainer.train_epoch(train_loader, val_loader, epoch)
    trainer.save_checkpoint(epoch, metrics)
    print(f"Epoch {epoch}: Loss={metrics['total_loss']:.4f}")
```

### Step 2: Save Model Checkpoints

```python
# Save best model
best_checkpoint_path = "/path/to/best_model.pth"
torch.save(vits.state_dict(), best_checkpoint_path)

# Save to cache (auto-load on startup)
cache_path = Path.home() / ".cache" / "kannada_tts"
torch.save(vits.state_dict(), cache_path / "vits_kannada.pth")
```

### Step 3: Update Model Manager

Edit `src/model_manager.py` to load from checkpoints:

```python
def _load_vits_from_checkpoint(self, checkpoint_path: str):
    """Load VITS from checkpoint file"""
    try:
        from src.hybrid.models import VITS
        
        # Load with same configuration as training
        model = VITS(
            num_chars=132,
            hidden_size=192,
            mel_channels=80,
            # ... other parameters from your training
        )
        
        # Load trained weights
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()  # Set to evaluation mode
        
        logger.info(f"✓ Loaded VITS from {checkpoint_path}")
        return model
    
    except Exception as e:
        logger.error(f"Failed to load VITS: {str(e)}")
        raise
```

### Step 4: Verify Model Load

Restart the application and check:

```bash
# Option 1: Via web UI
# Open http://localhost:8000 and test synthesis

# Option 2: Via API
curl http://localhost:8000/health
# Should show models_loaded: true

# Option 3: Check API docs  
# Open http://localhost:8000/docs and try endpoints
```

---

## Advanced Integration

### Custom Model Paths

Store models in a custom location:

```python
# In src/model_manager.py
def __init__(self, model_dir: str = None):
    if model_dir is None:
        self.model_cache_dir = Path.home() / ".cache" / "kannada_tts"
    else:
        self.model_cache_dir = Path(model_dir)
    
    self.model_cache_dir.mkdir(parents=True, exist_ok=True)
```

Use custom directory:

```python
# In app.py startup_event
from src.model_manager import ModelManager

# Use custom model directory
model_manager = ModelManager(model_dir="/path/to/my/models")
```

### Downloading Models from URL

```python
# In src/model_manager.py
def load_vits_model(self):
    model_path = self.model_cache_dir / "vits_kannada.pth"
    
    if not model_path.exists():
        # Download from your server
        try:
            self.download_model_from_url(
                "vits",
                "https://your-server.com/models/vits_kannada.pth"
            )
        except:
            # Fall back to default initialization
            logger.warning("Could not download model, using default")
            return self._initialize_vits_default()
    
    return self._load_vits_from_checkpoint(str(model_path))
```

### Multi-Model Support

Support multiple model variants:

```python
# In app.py
model_variants = {
    "small": "vits_kannada_small.pth",
    "medium": "vits_kannada_medium.pth",
    "large": "vits_kannada_large.pth"
}

@app.post("/api/synthesize")
async def synthesize(request: SynthesizeRequest):
    # request.model_variant = "small" / "medium" / "large"
    model_variant = request.model_variant or "medium"
    variant_path = model_manager.model_cache_dir / model_variants[model_variant]
    
    # Load appropriate model
    # ...
```

### Version Management

Track model versions:

```python
# Create model_versions.json
{
    "vits": {
        "version": "2.1",
        "date": "2026-03-01",
        "metrics": {
            "mcd": 3.8,
            "snr": 23.0
        }
    },
    "tacotron2": {
        "version": "1.0",
        "date": "2026-02-15",
        "metrics": {
            "mcd": 5.0,
            "snr": 21.0
        }
    }
}
```

Load specific versions:

```python
import json

def load_model_by_version(model_name: str, version: str):
    with open("model_versions.json") as f:
        versions = json.load(f)
    
    model_version = versions[model_name]
    if model_version["version"] == version:
        # Load this version
        return load_model(get_checkpoint_path(model_name, version))
```

---

## Model Update Workflow

### Scenario: New Training Iteration

1. **Train new model:**
   ```bash
   python train_vits.py --epochs 150 --lr 1e-4
   ```

2. **Save checkpoint:**
   ```python
   torch.save(vits.state_dict(), "vits_kannada_v2.pth")
   ```

3. **Backup old model:**
   ```bash
   mv ~/.cache/kannada_tts/vits_kannada.pth \
      ~/.cache/kannada_tts/vits_kannada_v1_backup.pth
   ```

4. **Install new model:**
   ```bash
   cp vits_kannada_v2.pth ~/.cache/kannada_tts/vits_kannada.pth
   ```

5. **Test:**
   ```bash
   python run_app.py
   # Visit http://localhost:8000 and test
   ```

6. **Roll back if needed:**
   ```bash
   mv ~/.cache/kannada_tts/vits_kannada_v1_backup.pth \
      ~/.cache/kannada_tts/vits_kannada.pth
   ```

---

## Troubleshooting

### Model Not Loading

**Check 1:** Verify model path exists
```bash
ls -la ~/.cache/kannada_tts/
```

**Check 2:** Verify model architecture matches initialization
```python
# Load and check state dict keys
state_dict = torch.load("your_model.pth", map_location="cpu")
print(state_dict.keys())
```

**Check 3:** Check model device
```python
# Ensure model is on correct device
model = model.to(device)
```

### Performance Issues After Update

1. Verify model is in eval mode: `model.eval()`
2. Check inference time: `time.time()` measurements
3. Compare metrics with baseline

### Memory Issues

If new model uses more memory:

```python
# Reduce model dimensions
# Or enable gradient checkpointing
from torch.utils.checkpoint import checkpoint
```

---

## API Integration

When using your custom models via API:

```bash
# Synthesize with current (your trained) model
curl -X POST "http://localhost:8000/api/synthesize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "ನಮಸ್ಕಾರ",
    "approach": "hybrid"
  }'

# Compare with traditional approach
curl -X POST "http://localhost:8000/api/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "ನಮಸ್ಕಾರ",
    "include_metrics": true
  }'
```

---

## Best Practices

1. **Always maintain backups** of previous model versions
2. **Test new models thoroughly** before deploying
3. **Track model versions** with metrics and dates
4. **Document model configurations** (hyperparameters, training data)
5. **Monitor performance** after updates
6. **Keep training reproducible** (save seeds, configs)
7. **Version control** model metadata but not binary weights

---

## Support

For issues with model integration:

1. Check [SETUP.md](SETUP.md) for installation guidance
2. Review [WEB_APP_README.md](WEB_APP_README.md) for API usage
3. Check application logs for error messages
4. Verify model compatibility with current PyTorch version

---

## Next Steps

1. Train your Kannada TTS models using the training interface
2. Save checkpoints to the cache directory
3. Restart the application to load your models
4. Test via the web UI or API
5. Monitor performance metrics
6. Deploy to production when satisfied
