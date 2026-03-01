# FastAPI Web Application - Setup & Deployment Guide

Complete guide for setting up and running the Kannada TTS web application with FastAPI.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Running the Application](#running-the-application)
4. [Configuration](#configuration)
5. [Adding Pre-trained Models](#adding-pre-trained-models)
6. [Deployment](#deployment)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- Python 3.8 or higher
- 2GB+ RAM (4GB recommended for GPU)
- Optional: NVIDIA GPU with CUDA 11.8+

### Python Packages
All required packages are listed in `requirements.txt`

### Operating Systems
- ✓ Linux (Ubuntu 18.04+)
- ✓ macOS (10.14+)
- ✓ Windows 10/11

---

## Installation

### 1. Clone or Download the Repository

```bash
cd kannada-tts
```

### 2. Create a Python Virtual Environment (Recommended)

**On Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python validate_setup.py
```

You should see:
```
✓ All checks passed! Ready to run the application.
```

---

## Running the Application

### Option 1: Using the Startup Script (Recommended)

```bash
python run_app.py
```

Output:
```
============================================================
Kannada Text-to-Speech System - Web Application
============================================================
Starting FastAPI server...
Open your browser and go to: http://localhost:8000

Endpoints:
  UI:        http://localhost:8000/
  API Docs:  http://localhost:8000/docs
  Health:    http://localhost:8000/health

Press Ctrl+C to stop the server
============================================================
```

### Option 2: Using Uvicorn Directly

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Option 3: Production Deployment

```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn + Uvicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app
```

---

## Configuration

### Port Configuration

To use a different port:

```bash
# Using run_app.py (edit the file)
# Change: port=8000 to your desired port

# Using uvicorn directly
uvicorn app:app --port 8001

# Using Gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:8001
```

### GPU/CPU Selection

Models automatically use GPU if available. To force CPU:

```python
# In app.py, modify the startup_event function to set device manually
# Change: device = "cuda" if torch.cuda.is_available() else "cpu"
# To: device = "cpu"
```

### Batch Size Adjustment

For systems with limited memory, reduce batch size:

```python
# In src/model_manager.py, modify VITS initialization
# Change batch_size during synthesis calls
```

---

## Adding Pre-trained Models

### Method 1: Download from URL

1. Place your model checkpoint in `~/.cache/kannada_tts/`:

```bash
# Create cache directory
mkdir -p ~/.cache/kannada_tts

# Copy your model files
cp your_vits_model.pth ~/.cache/kannada_tts/vits_kannada.pth
cp your_tacotron2_model.pth ~/.cache/kannada_tts/tacotron2_kannada.pth
```

2. On next application start, models will be automatically loaded.

### Method 2: Automatic Download

Update `src/model_manager.py` to download from a remote URL:

```python
def _initialize_vits_default(self):
    """Initialize VITS model with default configuration"""
    try:
        from src.hybrid.models import VITS
        
        # First try to download model
        try:
            self.download_model_from_url(
                "vits",
                "https://your-server.com/models/vits_kannada.pth"
            )
        except:
            # If download fails, initialize with default config
            pass
        
        vits = VITS(
            num_chars=132,
            hidden_size=192,
            # ... rest of configuration
        )
        
        return vits.to(self.device)
    except Exception as e:
        logger.error(f"Failed to initialize VITS: {str(e)}")
        raise
```

### Method 3: Using Model Manager API

```python
from src.model_manager import ModelManager

manager = ModelManager()

# Download model
manager.download_model_from_url(
    "vits",
    "https://your-server.com/models/vits_kannada.pth"
)

# Get model info
info = manager.get_model_info()
print(info)
```

---

## Deployment

### Docker Deployment

1. Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "run_app.py"]
```

2. Build image:

```bash
docker build -t kannada-tts:latest .
```

3. Run container:

```bash
docker run -p 8000:8000 kannada-tts:latest
```

### Kubernetes Deployment

Example `deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kannada-tts
spec:
  replicas: 2
  selector:
    matchLabels:
      app: kannada-tts
  template:
    metadata:
      labels:
        app: kannada-tts
    spec:
      containers:
      - name: kannada-tts
        image: kannada-tts:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: kannada-tts-service
spec:
  selector:
    app: kannada-tts
  type: LoadBalancer
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
```

### AWS Lambda Deployment

Use AWS Lambda with container images or serverless frameworks like Zappa.

### Heroku Deployment

1. Create `Procfile`:

```
web: gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app
```

2. Deploy:

```bash
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

---

## Troubleshooting

### Issue: Module not found

**Solution:**
```bash
# Ensure you're in the project root directory
cd kannada-tts

# Reinstall dependencies
pip install -r requirements.txt

# Run validation
python validate_setup.py
```

### Issue: CUDA Out of Memory

**Solution:**
```python
# In app.py, modify startup_event to force CPU
device = "cpu"  # Instead of checking cuda availability
```

### Issue: Port Already in Use

**Solution:**
```bash
# Find process using port
lsof -i :8000  # On Linux/macOS
netstat -ano | findstr :8000  # On Windows

# Kill process or use different port
uvicorn app:app --port 8001
```

### Issue: Models Not Loading

**Solution:**
```bash
# Check if models exist
ls ~/.cache/kannada_tts/

# Manually copy models
cp /path/to/model.pth ~/.cache/kannada_tts/

# Verify setup
python validate_setup.py
```

### Issue: Slow Inference

**Solutions:**
1. Use GPU instead of CPU
2. Batch multiple requests
3. Use smaller model variant
4. Enable caching in FastAPI

### Issue: High Memory Usage

**Solutions:**
```python
# In app.py, add memory management
import gc

@app.post("/api/synthesize")
async def synthesize(request: SynthesizeRequest):
    try:
        # ... synthesis code ...
        return result
    finally:
        gc.collect()  # Force garbage collection
        torch.cuda.empty_cache()  # Empty GPU cache
```

---

## Performance Optimization

### Enable Caching

```python
from fastapi_cache2 import FastAPICache2
from fastapi_cache2.backends.redis import RedisBackend
from aioredis import from_url

@app.on_event("startup")
async def startup():
    redis = await from_url("redis://localhost")
    FastAPICache2.init(RedisBackend(redis), prefix="fastapi-cache")
```

### Monitor Performance

```bash
# Install monitoring tools
pip install prometheus-client

# View metrics at http://localhost:8000/metrics
```

### Load Testing

```bash
# Install locust
pip install locust

# Create locustfile.py and run
locust -f locustfile.py
```

---

## Next Steps

1. **Test the application**: Open http://localhost:8000
2. **Try the UI**: Synthesize some Kannada text
3. **Compare approaches**: Use the comparison feature
4. **Check API docs**: Visit http://localhost:8000/docs
5. **Deploy to production**: Follow the deployment guide

## Support

For issues or questions:
1. Check [WEB_APP_README.md](WEB_APP_README.md)
2. Review [README.md](README.md)
3. Check [docs/](docs/) for detailed documentation
4. Run `python validate_setup.py` to diagnose problems

---

## License

See LICENSE file in the root directory.
