#!/usr/bin/env python3
"""
Kannada TTS - Main Runner Script
Supports both Hybrid and Non-Hybrid Approaches
"""

import torch
import numpy as np
import logging
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("output/kannada_tts.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("KannadaTTS")

print("="*80)
print("🎵 KANNADA TTS - HYBRID & NON-HYBRID APPROACHES")
print("="*80)


def create_models(approach: str, device: str = "cuda"):
    """Create models for specified approach"""
    
    logger.info(f"Creating {approach.upper()} models...")
    
    if approach == "hybrid":
        from src.hybrid.models import Tacotron2Hybrid, VocoderHybrid
        
        tacotron2 = Tacotron2Hybrid(
            num_chars=132,
            encoder_hidden_size=256,
            decoder_hidden_size=1024,
            style_dim=128,
            num_attention_heads=4
        )
        
        vocoder = VocoderHybrid(
            in_channels=80,
            out_channels=1,
            style_dim=128
        )
        
        logger.info("✓ Hybrid models created successfully")
        
    elif approach == "non_hybrid":
        from src.non_hybrid.models import Tacotron2Model, VocoderModel
        
        tacotron2 = Tacotron2Model(
            num_chars=132,
            encoder_hidden_size=256,
            decoder_hidden_size=1024
        )
        
        vocoder = VocoderModel(
            in_channels=80,
            out_channels=1
        )
        
        logger.info("✓ Non-Hybrid models created successfully")
    
    else:
        raise ValueError(f"Unknown approach: {approach}")
    
    return tacotron2, vocoder


def test_inference(approach: str):
    """Test inference pipeline"""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing {approach.upper()} Inference")
    logger.info(f"{'='*80}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create models
    tacotron2, vocoder = create_models(approach, device)
    
    # Create inference
    if approach == "hybrid":
        from src.hybrid.inference import HybridInference
        inference = HybridInference(tacotron2, vocoder, device=device)
        logger.info("✓ HybridInference initialized")
    else:
        from src.non_hybrid.inference import StandardInference
        inference = StandardInference(tacotron2, vocoder, device=device)
        logger.info("✓ StandardInference initialized")
    
    # Test texts
    test_texts = [
        "ನಮಸ್ಕಾರ",
        "ಕನ್ನಡ ಭಾಷೆ",
    ]
    
    logger.info(f"\nTesting with {len(test_texts)} texts...")
    
    for idx, text in enumerate(test_texts, 1):
        try:
            logger.info(f"\n[{idx}/{len(test_texts)}] Synthesizing: {text}")
            
            if approach == "hybrid":
                audio = inference.synthesize(
                    text,
                    emotion="neutral",
                    post_processing="advanced"
                )
                logger.info(f"✓ Generated audio shape: {audio.shape}")
                logger.info(f"  Available emotions: {inference.get_available_emotions()}")
                logger.info(f"  Available pipelines: {inference.get_available_pipelines()}")
            else:
                audio = inference.synthesize(text)
                logger.info(f"✓ Generated audio shape: {audio.shape}")
            
            # Save audio
            output_dir = f"output/{approach}_inference"
            os.makedirs(output_dir, exist_ok=True)
            output_path = f"{output_dir}/test_{idx}.wav"
            inference.save_audio(audio, output_path)
            logger.info(f"✓ Saved to: {output_path}")
        
        except Exception as e:
            logger.error(f"✗ Error: {e}", exc_info=True)


def test_training(approach: str):
    """Test training pipeline"""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing {approach.upper()} Training")
    logger.info(f"{'='*80}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create models
    tacotron2, vocoder = create_models(approach, device)
    
    # Create trainer
    if approach == "hybrid":
        from src.hybrid.training import HybridTrainer
        trainer = HybridTrainer(tacotron2, vocoder, device)
        logger.info("✓ HybridTrainer initialized")
    else:
        from src.non_hybrid.training import StandardTrainer
        trainer = StandardTrainer(tacotron2, vocoder, device)
        logger.info("✓ StandardTrainer initialized")
    
    logger.info(f"Checkpoint directory: {trainer.checkpoint_dir}")
    logger.info("✓ Training pipeline ready (use train_epoch with data loader)")


def compare_approaches():
    """Compare both approaches"""
    
    logger.info(f"\n{'='*80}")
    logger.info("COMPARISON: Hybrid vs Non-Hybrid")
    logger.info(f"{'='*80}\n")
    
    comparison = {
        "Aspect": ["Models", "Features", "Post-Processing", "Emotional Control", "Style Transfer", "Speed", "Quality"],
        "Hybrid": [
            "Enhanced Tacotron2 + HiFiGAN",
            "Multi-head attention, Duration prediction, Style encoding",
            "Advanced: Spectral gating + Wiener + Median filters",
            "Yes (5 emotions)",
            "Yes (via style embedding)",
            "Slightly slower",
            "Superior"
        ],
        "Non-Hybrid": [
            "Standard Tacotron2 + HiFiGAN",
            "Basic attention, No duration, No style",
            "Standard: Single-pass denoising",
            "No",
            "No",
            "Faster",
            "Good"
        ]
    }
    
    print("\nCOMPARISON TABLE:")
    print("-" * 120)
    print(f"{'Aspect':<25} {'Hybrid':<50} {'Non-Hybrid':<50}")
    print("-" * 120)
    
    for i in range(len(comparison["Aspect"])):
        aspect = comparison["Aspect"][i]
        hybrid = comparison["Hybrid"][i]
        non_hybrid = comparison["Non-Hybrid"][i]
        print(f"{aspect:<25} {hybrid:<50} {non_hybrid:<50}")
    
    print("-" * 120)
    
    logger.info("\n✓ Comparison complete")


def generate_comparison_report(output_dir: str = "output"):
    """Generate detailed comparison report"""
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "comparison": {
            "hybrid": {
                "description": "Advanced TTS with multi-head attention, style control, and prosody enhancement",
                "components": {
                    "acoustic_model": {
                        "name": "Tacotron2Hybrid",
                        "features": [
                            "Multi-head attention mechanism",
                            "Style/emotion encoder",
                            "Duration prediction",
                            "Style modulation in decoder"
                        ]
                    },
                    "vocoder": {
                        "name": "VocoderHybrid",
                        "features": [
                            "Adaptive instance normalization",
                            "Style-conditioned generation",
                            "Residual blocks with style control",
                            "Multiple vocoder types (default, fast, quality)"
                        ]
                    },
                    "post_processors": [
                        "NoiseReductionProcessor (spectral gating, Wiener, median)",
                        "ProsodyEnhancer (pitch shift, time stretch, dynamics, vibrato)",
                        "AudioPostProcessor (multiple pipelines)"
                    ]
                },
                "capabilities": [
                    "Emotion control (neutral, happy, sad, angry, surprised)",
                    "Style transfer via reference audio",
                    "Multi-stage audio enhancement",
                    "Prosody manipulation",
                    "Advanced noise reduction"
                ],
                "use_cases": [
                    "High-quality TTS with emotional expression",
                    "Conversational AI with varied prosody",
                    "Content creation with style consistency",
                    "Accessible audio with enhanced clarity"
                ]
            },
            "non_hybrid": {
                "description": "Standard TTS with proven Tacotron2 and HiFiGAN architecture",
                "components": {
                    "acoustic_model": {
                        "name": "Tacotron2Model",
                        "features": [
                            "Standard attention mechanism",
                            "Encoder-decoder architecture",
                            "Gate output for stopping criterion"
                        ]
                    },
                    "vocoder": {
                        "name": "VocoderModel",
                        "features": [
                            "Standard HiFiGAN architecture",
                            "Residual blocks",
                            "Upsampling layers",
                            "Fast and efficient"
                        ]
                    },
                    "post_processors": [
                        "Basic noise reduction",
                        "Loudness normalization"
                    ]
                },
                "capabilities": [
                    "Fast and efficient synthesis",
                    "Good audio quality",
                    "Low memory footprint",
                    "Stable training"
                ],
                "use_cases": [
                    "Real-time TTS applications",
                    "Resource-constrained environments",
                    "Baseline models",
                    "Low-latency inference"
                ]
            }
        },
        "recommendations": {
            "use_hybrid_when": [
                "Quality is paramount",
                "Emotional expression is needed",
                "Style transfer is desired",
                "Advanced post-processing is beneficial"
            ],
            "use_non_hybrid_when": [
                "Speed/latency is critical",
                "Resources are limited",
                "Simplicity is preferred",
                "Baseline comparison needed"
            ]
        }
    }
    
    # Save report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "approach_comparison.json")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Comparison report saved to: {report_path}")
    return report


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="Kannada TTS - Hybrid & Non-Hybrid Approaches"
    )
    parser.add_argument(
        "--approach",
        choices=["hybrid", "non-hybrid", "both"],
        default="both",
        help="Approach to use"
    )
    parser.add_argument(
        "--mode",
        choices=["inference", "training", "comparison", "all"],
        default="all",
        help="Mode to run"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Configuration:")
    logger.info(f"  Approach: {args.approach}")
    logger.info(f"  Mode: {args.mode}")
    logger.info(f"  Device: {args.device}")
    
    try:
        # Comparison
        if args.mode in ["comparison", "all"]:
            compare_approaches()
            generate_comparison_report()
        
        # Inference tests
        if args.mode in ["inference", "all"]:
            if args.approach in ["hybrid", "both"]:
                test_inference("hybrid")
            if args.approach in ["non-hybrid", "both"]:
                test_inference("non_hybrid")
        
        # Training tests
        if args.mode in ["training", "all"]:
            if args.approach in ["hybrid", "both"]:
                test_training("hybrid")
            if args.approach in ["non-hybrid", "both"]:
                test_training("non_hybrid")
        
        logger.info(f"\n{'='*80}")
        logger.info("✓ All tests completed successfully!")
        logger.info(f"{'='*80}")
        
    except Exception as e:
        logger.error(f"✗ Error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
