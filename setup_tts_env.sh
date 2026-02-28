#!/bin/bash
# Kannada TTS Environment Setup (Colab/Linux VM)
# Run: chmod +x setup_tts_env.sh && ./setup_tts_env.sh

set -e  # Exit on any error

echo "Setting up Kannada TTS Environment..."

# 1. System dependencies
sudo apt-get update -y
sudo apt-get install -y \
  python3.10 \
  python3.10-venv \
  python3.10-dev \
  sox \
  ffmpeg \
  espeak-ng \
  libsndfile1 \
  build-essential

# 2. Create Python 3.10 virtual environment
echo "Creating Python 3.10 venv..."
python3.10 -m venv /content/py310
source /content/py310/bin/activate

# 3. Upgrade pip tools
pip install --upgrade pip setuptools wheel

# 4. Install PyTorch with CUDA 12.1
echo "Installing PyTorch CUDA..."
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

# 5. Install Coqui TTS (stable VITS)
echo "Installing Coqui TTS..."
pip install TTS

# 6. Audio + text processing libraries
pip install \
  librosa \
  soundfile \
  numpy \
  pandas \
  tqdm \
  scipy \
  indic-transliteration \
  unidecode \
  kagglehub

# 7. Verify installation
echo "Verifying installation..."
python --version
tts --help

echo "SETUP COMPLETE!"
echo "Activate env: source /content/py310/bin/activate"
echo "Run TTS:      python src/inference.py"
