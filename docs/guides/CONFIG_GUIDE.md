Configuration Reference

VITS MODEL PARAMETERS
====================

Core Settings:
  num_chars: 132                  # Kannada character vocabulary size
  hidden_size: 192                # Model hidden dimension
  mel_channels: 80                # Mel-spectrogram channels
  sample_rate: 22050              # Audio sample rate (Hz)
  n_fft: 1024                     # FFT size
  hop_length: 256                 # STFT hop length
  f_min: 40                       # Minimum frequency (Hz)
  f_max: 7600                     # Maximum frequency (Hz)

VAE Settings:
  kl_weight: 1.0                  # KL divergence loss weight
  mel_loss_weight: 45.0           # Mel reconstruction weight
  duration_loss_weight: 0.1       # Duration prediction weight

TRAINING PARAMETERS
===================

Optimizer:
  type: Adam
  learning_rate: 1e-4
  betas: [0.9, 0.999]
  eps: 1e-9

Scheduler:
  type: ExponentialLR
  gamma: 0.99                     # Decay rate

Gradient Clipping:
  max_norm: 1.0                   # Gradient norm clipping

Training Loop:
  epochs: 100                     # Number of epochs
  batch_size: 16                  # Training batch size
  eval_batch_size: 8              # Evaluation batch size
  save_step: 10                   # Save checkpoint every N epochs
  val_step: 1                     # Validate every N epochs

Data:
  num_workers: 4                  # Data loader workers
  pin_memory: true                # Pin GPU memory for faster transfer

INFERENCE PARAMETERS
====================

Synthesis:
  temperature: 0.667              # VAE latent temperature (variability control)
  max_mel_steps: 1000             # Maximum mel-spectrogram length
  gate_threshold: 0.5             # Gate threshold for stopping

Griffin-Lim Vocoding (Fallback):
  n_fft: 1024
  hop_length: 256
  n_iters: 100                    # Griffin-Lim iterations

AUDIO PROCESSING PARAMETERS
===========================

Noise Reduction:
  method: spectral_gating         # or "wiener"
  threshold_db: -40               # Spectral gating threshold
  noise_profile_duration: 0.5     # Duration in seconds

Prosody Enhancement:
  emotion_params:
    neutral:
      pitch_shift: 0.0            # Semitones
      duration_scale: 1.0
      energy_scale: 1.0
    
    happy:
      pitch_shift: 2.0
      duration_scale: 0.9
      energy_scale: 1.2
    
    sad:
      pitch_shift: -1.5
      duration_scale: 1.2
      energy_scale: 0.8
    
    angry:
      pitch_shift: 1.0
      duration_scale: 0.8
      energy_scale: 1.4
    
    calm:
      pitch_shift: -0.5
      duration_scale: 1.1
      energy_scale: 0.9

Post-Processing Pipelines:
  none:
    No processing applied
  
  basic:
    - Light noise reduction
  
  advanced:
    - Noise reduction
    - Prosody enhancement
    - Post-processing
  
  quality:
    - Heavy noise reduction
    - Advanced post-processing
  
  speed:
    - Minimal processing
    - Fast inference

GPU MEMORY OPTIMIZATION
=======================

For GPUs with Limited Memory:

  Base Configuration (8GB GPU):
    num_chars: 132
    hidden_size: 128
    batch_size: 4
    eval_batch_size: 2
    num_workers: 0

  Medium Configuration (16GB GPU):
    hidden_size: 192
    batch_size: 16
    eval_batch_size: 8
    num_workers: 4

  High-End Configuration (24GB GPU):
    hidden_size: 256
    batch_size: 32
    eval_batch_size: 16
    num_workers: 8

RECOMMENDED CONFIGURATIONS
==========================

For Development:
  hidden_size: 128
  batch_size: 8
  epochs: 10
  device: cpu

For Training on GPU:
  hidden_size: 192
  batch_size: 16
  epochs: 100
  learning_rate: 1e-4
  device: cuda

For High Quality:
  hidden_size: 256
  batch_size: 32
  epochs: 500
  learning_rate: 5e-5
  kl_weight: 0.5

For Fast Inference:
  hidden_size: 128
  temperature: 0.3
  post_processing: "speed"

For Production:
  hidden_size: 192
  batch_size: 16
  post_processing: "advanced"
  gradient_clip: 1.0

TUNING GUIDELINES
================

If Training Too Slow:
  - Reduce hidden_size to 128
  - Reduce batch_size to 8
  - Use num_workers: 4
  - Enable mixed precision
  - Use gradient accumulation

If Running Out of Memory:
  - Reduce batch_size
  - Reduce hidden_size
  - Set num_workers: 0
  - Use device: cpu
  - Enable gradient checkpointing

If Audio Quality Poor:
  - Increase hidden_size to 256
  - Train more epochs (200+)
  - Reduce learning_rate to 5e-5
  - Increase mel_loss_weight to 50-100
  - Use post_processing: "quality"

If Training Unstable:
  - Reduce learning_rate
  - Increase gradient_clip to 2.0
  - Reduce batch_size
  - Start with lower kl_weight (0.1-0.5)
  - Check data normalization

If Inference Too Slow:
  - Use smaller hidden_size (128)
  - Batch process multiple utterances
  - Use post_processing: "speed"
  - Profile to find bottleneck
  - Consider quantization

If KL Loss Too High (Posterior Collapse):
  - Reduce kl_weight to 0.1-0.5
  - Increase model capacity
  - Use KL annealing schedule
  - Verify data normalization

CHECKPOINT MANAGEMENT
====================

Checkpoint Format:
  {
    'epoch': int,
    'model_state': dict,
    'optimizer_state': dict,
    'scheduler_state': dict,
    'metrics': dict
  }

Saving Checkpoints:
  trainer.save_checkpoint(epoch, metrics)
  Saves to: output/vits_checkpoints/checkpoint_epoch_X.pt

Loading Checkpoints:
  epoch, metrics = trainer.load_checkpoint(path)

Best Practice:
  - Save every 10-50 epochs
  - Keep 3-5 most recent checkpoints
  - Keep separate best_model.pt checkpoint

ENVIRONMENT VARIABLES
====================

Optional settings:

OMP_NUM_THREADS: Number of CPU threads
  Default: 4
  Adjust based on CPU cores

CUDA_DEVICE_ORDER: GPU device ordering
  Default: PCI_BUS_ID

CUDA_VISIBLE_DEVICES: Restrict GPU usage
  Example: 0,1 (use GPUs 0 and 1)

---

Version: 2.0
Last Updated: 2026-02-28
Status: Production
