API Reference

VITS MODEL API
==============

VITS Class

Constructor:
  VITS(num_chars=132, hidden_size=192, mel_channels=80, kl_weight=1.0)
  
  Parameters:
    num_chars (int): Number of characters in vocabulary (default: 132 for Kannada)
    hidden_size (int): Hidden dimension size (default: 192)
    mel_channels (int): Number of mel-spectrogram channels (default: 80)
    kl_weight (float): Weight for KL divergence loss (default: 1.0)

Methods:
  forward(text_input, text_lengths, mels=None):
    Perform forward pass through VITS
    
    Args:
      text_input (Tensor): Character indices (batch_size, max_text_length)
      text_lengths (Tensor): Length of each text sequence (batch_size)
      mels (Tensor, optional): Target mel-spectrograms for training (batch_size, mel_length, 80)
    
    Returns:
      dict containing:
        mel_output: Predicted mel-spectrograms
        z: Sampled latent codes
        mean: Posterior mean
        logstd: Posterior log-std
        durations: Predicted durations
    
    Note: When mels provided (training mode), samples z from posterior
          When mels is None (inference mode), samples z from prior N(0, I)

Properties:
  text_encoder: TextEncoder component
  posterior_encoder: PosteriorEncoder component
  duration_predictor: DurationPredictor component
  generator: Generator component

Save/Load:
  torch.save(vits.state_dict(), path)
  vits.load_state_dict(torch.load(path))

---

VITS INFERENCE API
==================

VITSInference Class

Constructor:
  VITSInference(vits_model, device="cuda", vocoder=None)
  
  Parameters:
    vits_model (VITS): Trained VITS model
    device (str): Device to use ("cuda" or "cpu")
    vocoder (callable, optional): Custom vocoder function

Methods:
  synthesize(text, temperature=0.667, emotion="neutral", post_processing="advanced"):
    Synthesize speech from text
    
    Args:
      text (str): Input text in Kannada
      temperature (float): VAE sampling temperature (default: 0.667)
                          Higher = more variability
                          Lower = more deterministic
      emotion (str): Emotion type: "neutral", "happy", "sad", "angry", "calm"
      post_processing (str): Processing mode: "none", "basic", "advanced"
    
    Returns:
      audio (numpy array): Audio waveform (22050 Hz, mono)
    
    Raises:
      ValueError: If text contains unsupported characters
  
  synthesize_batch(texts, temperature=0.667):
    Synthesize multiple utterances
    
    Args:
      texts (list): List of text strings
      temperature (float): VAE sampling temperature
    
    Returns:
      audios (list): List of audio arrays
  
  text_to_sequence(text):
    Convert text to character sequence
    
    Args:
      text (str): Input text
    
    Returns:
      tuple: (sequence tensor, length)
  
  save_audio(audio, output_path, sample_rate=22050):
    Save audio to WAV file
    
    Args:
      audio (numpy array): Audio waveform
      output_path (str): Output file path
      sample_rate (int): Sample rate (default: 22050)

Properties:
  char_map: Dictionary mapping characters to indices
  device: Current device
  model: VITS model

---

VITS TRAINER API
================

VITSTrainer Class

Constructor:
  VITSTrainer(vits_model, device="cuda")
  
  Parameters:
    vits_model (VITS): VITS model to train
    device (str): Device to use ("cuda" or "cpu")

Methods:
  train_epoch(train_loader, val_loader, epoch):
    Train for one epoch
    
    Args:
      train_loader (DataLoader): Training data loader
      val_loader (DataLoader): Validation data loader
      epoch (int): Epoch number
    
    Returns:
      dict: Metrics dictionary containing:
        train_mel_loss, train_kl_loss, train_duration_loss, train_total_loss
        val_mel_loss, val_kl_loss, val_duration_loss, val_total_loss
  
  train_step(text_input, text_lengths, target_mels, target_durations):
    Single training step
    
    Args:
      text_input (Tensor): Character indices (batch_size, max_text_length)
      text_lengths (Tensor): Text lengths (batch_size)
      target_mels (Tensor): Target mel-spectrograms (batch_size, mel_length, 80)
      target_durations (Tensor): Target durations (batch_size, text_length)
    
    Returns:
      dict: Step losses
  
  validate(val_loader):
    Validation step
    
    Args:
      val_loader (DataLoader): Validation data loader
    
    Returns:
      dict: Validation metrics
  
  save_checkpoint(epoch, metrics):
    Save checkpoint
    
    Args:
      epoch (int): Epoch number
      metrics (dict): Metrics to save
  
  load_checkpoint(checkpoint_path):
    Load checkpoint
    
    Args:
      checkpoint_path (str): Path to checkpoint file
    
    Returns:
      tuple: (epoch, metrics) or (0, {}) if not found

Properties:
  optimizer: Adam optimizer
  scheduler: Learning rate scheduler
  checkpoint_dir: Directory for saving checkpoints

---

UNIFIED INFERENCE API
=====================

TTSInference Class

Constructor:
  TTSInference(approach="hybrid", model_type="vits", device="cuda")
  
  Parameters:
    approach (str): "hybrid" or "non_hybrid"
    model_type (str): "vits" (for hybrid) or "tacotron2" (for non_hybrid)
    device (str): "cuda" or "cpu"

Methods:
  synthesize(text, emotion="neutral", post_processing="advanced"):
    Synthesize speech
    
    Args:
      text (str): Input text
      emotion (str): Emotion type
      post_processing (str): Processing mode
    
    Returns:
      audio (numpy array): Audio waveform
  
  synthesize_batch(texts):
    Batch synthesis
    
    Args:
      texts (list): List of text strings
    
    Returns:
      audios (list): List of audio arrays

---

UNIFIED TRAINER API
===================

TTSTrainer Class

Constructor:
  TTSTrainer(approach="hybrid", model_type="vits", device="cuda")
  
  Parameters:
    approach (str): "hybrid" or "non_hybrid"
    model_type (str): "vits" or "tacotron2"
    device (str): "cuda" or "cpu"

Methods:
  train_epoch(train_loader, val_loader, epoch):
    Train one epoch, returns metrics dict

---

AUDIO PROCESSORS API
====================

NoiseReductionProcessor Class

Constructor:
  NoiseReductionProcessor(sr=22050)

Methods:
  denoise(audio, method="spectral_gating"):
    Remove noise from audio
    
    Args:
      audio (numpy array): Input audio
      method (str): "spectral_gating" or "wiener"
    
    Returns:
      audio (numpy array): Denoised audio
  
  denoise_advanced(audio):
    Multi-stage denoising
    
    Args:
      audio (numpy array): Input audio
    
    Returns:
      audio (numpy array): Processed audio

---

ProsodyEnhancer Class

Constructor:
  ProsodyEnhancer(sr=22050)

Methods:
  pitch_shift(audio, shift_factor):
    Shift pitch
    
    Args:
      audio (numpy array): Input audio
      shift_factor (float): Pitch shift (1.1 = 10% higher)
    
    Returns:
      audio (numpy array): Pitch-shifted audio
  
  time_stretch(audio, duration_factor):
    Change speed without changing pitch
    
    Args:
      audio (numpy array): Input audio
      duration_factor (float): Duration multiplier (0.9 = 10% faster)
    
    Returns:
      audio (numpy array): Time-stretched audio
  
  emotion_control(audio, emotion):
    Apply emotion processing
    
    Args:
      audio (numpy array): Input audio
      emotion (str): "neutral", "happy", "sad", "angry", "calm"
    
    Returns:
      audio (numpy array): Emotion-processed audio

---

AudioPostProcessor Class

Constructor:
  AudioPostProcessor()

Methods:
  process(audio, pipeline="advanced", emotion="neutral"):
    Post-process audio
    
    Args:
      audio (numpy array): Input audio
      pipeline (str): "standard", "advanced", "quality", "speed"
      emotion (str): Emotion type
    
    Returns:
      audio (numpy array): Processed audio

---

DATA FORMATS
============

Audio:
  - Sample Rate: 22050 Hz
  - Channels: Mono (1)
  - Bit Depth: Float32
  - Range: [-1.0, 1.0]

Mel-spectrogram:
  - Channels: 80
  - F-min: 40 Hz
  - F-max: 7600 Hz
  - Hop Length: 256
  - Window: Hann

Text:
  - Kannada characters (U+0C80 to U+0CFF)
  - 132 character vocabulary
  - No punctuation needed (handled internally)

---

ERROR HANDLING
==============

Common Exceptions:

ValueError: Text contains unsupported characters
  Solution: Use only Kannada characters (U+0C80 to U+0CFF)

RuntimeError: CUDA out of memory
  Solution: Reduce batch_size, model size, or use device="cpu"

FileNotFoundError: Model checkpoint not found
  Solution: Verify path and train model first

TypeError: Input tensor shape mismatch
  Solution: Check input dimensions and batch size

---

CONSTANTS
=========

Kannada Characters: 132 total
  - Vowels: 14
  - Consonants: 36
  - Vowel Signs: 16
  - Length Marks: 2
  - Sign Nukta: 1
  - Sign Virama: 1
  - Other Marks: 4
  - Punctuation/Special: 8

Default Hyperparameters:
  VITS Hidden Size: 192
  Mel Channels: 80
  Sample Rate: 22050 Hz
  Learning Rate: 1e-4
  Batch Size: 16
  KL Weight: 1.0
  Duration Loss Weight: 0.1

---

Version: 2.0
Last Updated: 2026-02-28
