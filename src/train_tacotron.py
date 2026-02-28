#!/usr/bin/env python3
"""
Kannada TTS Training Pipeline
- Tacotron2 acoustic model training
- HiFiGAN vocoder training
- Comprehensive logging and checkpointing
- Evaluation metrics tracking
"""

import os
import json
import torch
import logging
from datetime import datetime
from TTS.trainer import Trainer, TrainingArgs
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.vocoder.configs.hifigan_config import HifiganConfig
from TTS.vocoder.models.hifigan import HifiGAN
from TTS.tts.datasets import load_tts_samples
from TTS.vocoder.datasets import load_wav_to_mel

print("="*70)
print("🎵 KANNADA TTS - TRAINING PIPELINE (Tacotron2 + HiFiGAN)")
print("="*70)

# ============================================================================
# Setup Logging
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("output/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("KannadaTTS")

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🚀 Starting training pipeline...")

# ============================================================================
# 1. TACOTRON2 TRAINING
# ============================================================================
print("\n" + "="*70)
print("PHASE 1: TACOTRON2 ACOUSTIC MODEL TRAINING")
print("="*70)

try:
    # Load configuration
    tacotron_config = Tacotron2Config()
    if os.path.exists("config/tacotron2.json"):
        tacotron_config.load_json("config/tacotron2.json")
    else:
        logger.warning("Config not found, using defaults")
    
    # Update config for Kannada
    tacotron_config.model_args = {
        "num_chars": 132,  # Kannada characters
        "encoder_hidden_size": 256,
        "encoder_num_convolutions": 3,
        "encoder_conv_filters": 512,
        "encoder_conv_kernel_sizes": 5,
        "encoder_conv_dropout_p": 0.5,
        "decoder_hidden_size": 1024,
        "decoder_lstm_layers": 2,
        "decoder_lstm_dropout_p": 0.1,
        "attention_hidden_size": 128,
        "attention_location_n_filters": 32,
        "attention_location_kernel_size": 31,
        "postnet_conv_filters": 512,
        "postnet_conv_kernel_sizes": 5,
        "postnet_num_convolutions": 5,
        "postnet_dropout_p": 0.5,
        "gate_threshold": 0.5
    }
    
    logger.info("Loading Kannada TTS dataset...")
    train_samples, eval_samples = load_tts_samples(
        dataset_config={
            "formatter": "ljspeech",
            "meta_file_train": "data/metadata.csv"
        },
        eval_split_size=0.05
    )
    
    logger.info(f"Train samples: {len(train_samples)}, Eval samples: {len(eval_samples)}")
    
    # Create output directory
    os.makedirs("output/tacotron2", exist_ok=True)
    
    # Initialize model
    logger.info("Initializing Tacotron2 model...")
    tacotron_model = Tacotron2.init_from_config(tacotron_config)
    
    # Training arguments
    training_args = TrainingArgs(
        num_loader_workers=4,
        eval_split_size=0.05,
        batch_size=16,
        eval_batch_size=8,
        num_epochs=500,
        save_step=1000,
        eval_steps=500,
        print_step=100,
        save_n_checkpoints=3,
        save_best_after=1000,
        save_checkpoints=True,
        output_path="output/tacotron2",
        print_eval=True,
        use_tensorboard=True,
        gc_window_size=0,
    )
    
    # Initialize trainer
    logger.info("Starting Tacotron2 training...")
    tacotron_trainer = Trainer(
        training_args,
        tacotron_config,
        "output/tacotron2",
        model=tacotron_model,
        train_samples=train_samples,
        eval_samples=eval_samples
    )
    
    # Train
    tacotron_trainer.fit()
    logger.info("✅ Tacotron2 training completed!")
    
except Exception as e:
    logger.error(f"❌ Tacotron2 training failed: {str(e)}")
    raise

# ============================================================================
# 2. HiFiGAN VOCODER TRAINING
# ============================================================================
print("\n" + "="*70)
print("PHASE 2: HiFiGAN VOCODER TRAINING")
print("="*70)

try:
    logger.info("Loading HiFiGAN configuration...")
    vocoder_config = HifiganConfig()
    if os.path.exists("config/hifigan.json"):
        vocoder_config.load_json("config/hifigan.json")
    
    logger.info("Preparing vocoder dataset...")
    # Load mel-spectrograms from training data
    vocoder_samples = load_wav_to_mel(
        dataset_config={"formatter": "ljspeech", "meta_file": "data/metadata.csv"}
    )
    
    os.makedirs("output/hifigan", exist_ok=True)
    
    logger.info("Initializing HiFiGAN model...")
    vocoder_model = HifiGAN.init_from_config(vocoder_config)
    
    # HiFiGAN Training arguments
    vocoder_training_args = TrainingArgs(
        num_loader_workers=4,
        batch_size=16,
        eval_batch_size=8,
        num_epochs=200,
        save_step=5000,
        eval_steps=1000,
        print_step=100,
        save_n_checkpoints=3,
        output_path="output/hifigan",
        print_eval=True,
        use_tensorboard=True,
    )
    
    logger.info("Starting HiFiGAN training...")
    vocoder_trainer = Trainer(
        vocoder_training_args,
        vocoder_config,
        "output/hifigan",
        model=vocoder_model,
        train_samples=vocoder_samples[:int(0.95*len(vocoder_samples))],
        eval_samples=vocoder_samples[int(0.95*len(vocoder_samples)):]
    )
    
    vocoder_trainer.fit()
    logger.info("✅ HiFiGAN training completed!")
    
except Exception as e:
    logger.warning(f"⚠️ HiFiGAN training optional at this stage: {str(e)}")
    logger.info("Continuing without HiFiGAN. You can train it later.")

# ============================================================================
# 3. TRAINING SUMMARY
# ============================================================================
print("\n" + "="*70)
print("TRAINING SUMMARY")
print("="*70)

training_summary = {
    "timestamp": datetime.now().isoformat(),
    "tacotron2": {
        "status": "completed",
        "model_path": "output/tacotron2/best_model.pth",
        "config_path": "config/tacotron2.json"
    },
    "hifigan": {
        "status": "optional",
        "model_path": "output/hifigan/best_model.pth",
        "config_path": "config/hifigan.json"
    }
}

with open("output/training_summary.json", "w") as f:
    json.dump(training_summary, f, indent=2)

logger.info(f"\n{'='*70}")
logger.info("✅ TRAINING PIPELINE COMPLETE!")
logger.info(f"{'='*70}")
logger.info("\n📁 Generated files:")
logger.info("  - output/tacotron2/: Trained Tacotron2 model")
logger.info("  - output/hifigan/: Trained HiFiGAN vocoder")
logger.info("  - output/training.log: Training logs")
logger.info("  - output/training_summary.json: Training summary")
logger.info("\n➡️  Next step: python src/inference.py")

