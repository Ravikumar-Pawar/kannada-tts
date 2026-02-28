!sudo apt-get update -y
!sudo apt-get install -y \
  python3.10 \
  python3.10-venv \
  python3.10-dev \
  sox \
  ffmpeg \
  espeak-ng \
  libsndfile1 \
  build-essential

# Create Python 3.10 virtual environment
!python3.10 -m venv /content/py310

# Upgrade pip tools inside venv
!/content/py310/bin/pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA (Colab GPU)
!/content/py310/bin/pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

# Install Coqui TTS (VITS – stable)
!/content/py310/bin/pip install TTS

# Install audio + text processing libs
!/content/py310/bin/pip install \
  librosa \
  soundfile \
  numpy \
  pandas \
  tqdm \
  scipy \
  indic-transliteration \
  unidecode

# Verify installation
!/content/py310/bin/python --version
!/content/py310/bin/tts --help
