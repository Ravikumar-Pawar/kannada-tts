#!/usr/bin/env python3
"""
Kannada TTS Examples - Demonstrating Both Approaches
Complete examples for hybrid and non-hybrid TTS systems
"""

import logging
import os

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# EXAMPLE 1: Basic Hybrid Inference
# ==============================================================================

def example_1_hybrid_basic():
    """Example 1: Basic hybrid inference"""
    
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Hybrid Inference")
    print("="*80)
    
    from src.hybrid.models import Tacotron2Hybrid, VocoderHybrid
    from src.hybrid.inference import HybridInference
    
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create models
    print("\n1. Creating models...")
    tacotron2 = Tacotron2Hybrid(num_chars=132, style_dim=128)
    vocoder = VocoderHybrid(style_dim=128)
    print("   ✓ Models created")
    
    # Create inference
    print("\n2. Initializing inference...")
    inference = HybridInference(tacotron2, vocoder, device=device)
    print("   ✓ Inference ready")
    
    # Synthesize
    print("\n3. Synthesizing text...")
    text = "ನಮಸ್ಕಾರ"
    audio = inference.synthesize(text, emotion="neutral", post_processing="standard")
    print(f"   ✓ Generated audio: {audio.shape}")
    
    # Save
    print("\n4. Saving audio...")
    os.makedirs("output/examples", exist_ok=True)
    inference.save_audio(audio, "output/examples/example_1_hybrid_basic.wav")
    print("   ✓ Saved: output/examples/example_1_hybrid_basic.wav")


# ==============================================================================
# EXAMPLE 2: Non-Hybrid Basic Inference
# ==============================================================================

def example_2_non_hybrid_basic():
    """Example 2: Basic non-hybrid inference"""
    
    print("\n" + "="*80)
    print("EXAMPLE 2: Basic Non-Hybrid Inference")
    print("="*80)
    
    from src.non_hybrid.models import Tacotron2Model, VocoderModel
    from src.non_hybrid.inference import StandardInference
    
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create models
    print("\n1. Creating models...")
    tacotron2 = Tacotron2Model(num_chars=132)
    vocoder = VocoderModel()
    print("   ✓ Models created")
    
    # Create inference
    print("\n2. Initializing inference...")
    inference = StandardInference(tacotron2, vocoder, device=device)
    print("   ✓ Inference ready")
    
    # Synthesize
    print("\n3. Synthesizing text...")
    text = "ನಮಸ್ಕಾರ"
    audio = inference.synthesize(text)
    print(f"   ✓ Generated audio: {audio.shape}")
    
    # Save
    print("\n4. Saving audio...")
    os.makedirs("output/examples", exist_ok=True)
    inference.save_audio(audio, "output/examples/example_2_non_hybrid_basic.wav")
    print("   ✓ Saved: output/examples/example_2_non_hybrid_basic.wav")


# ==============================================================================
# EXAMPLE 3: Hybrid with Emotion Control
# ==============================================================================

def example_3_hybrid_emotions():
    """Example 3: Hybrid inference with different emotions"""
    
    print("\n" + "="*80)
    print("EXAMPLE 3: Hybrid Inference - Emotion Control")
    print("="*80)
    
    from src.hybrid.models import Tacotron2Hybrid, VocoderHybrid
    from src.hybrid.inference import HybridInference
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create models and inference
    print("\n1. Creating models and inference...")
    tacotron2 = Tacotron2Hybrid(num_chars=132, style_dim=128)
    vocoder = VocoderHybrid(style_dim=128)
    inference = HybridInference(tacotron2, vocoder, device=device)
    print("   ✓ Ready")
    
    # Test different emotions
    text = "ಕನ್ನಡ ಭಾಷೆ"
    emotions = ["neutral", "happy", "sad", "angry", "surprised"]
    
    print(f"\n2. Synthesizing '{text}' with different emotions...")
    
    os.makedirs("output/examples/emotions", exist_ok=True)
    
    for emotion in emotions:
        print(f"   - Generating {emotion}...")
        audio = inference.synthesize(
            text,
            emotion=emotion,
            post_processing="advanced"
        )
        output_path = f"output/examples/emotions/example_3_{emotion}.wav"
        inference.save_audio(audio, output_path)
        print(f"     ✓ Saved: {output_path}")
    
    print("\n✓ All emotion examples generated")


# ==============================================================================
# EXAMPLE 4: Hybrid with Post-Processing Pipelines
# ==============================================================================

def example_4_hybrid_pipelines():
    """Example 4: Different post-processing pipelines"""
    
    print("\n" + "="*80)
    print("EXAMPLE 4: Hybrid Inference - Post-Processing Pipelines")
    print("="*80)
    
    from src.hybrid.models import Tacotron2Hybrid, VocoderHybrid
    from src.hybrid.inference import HybridInference
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create models and inference
    print("\n1. Creating models and inference...")
    tacotron2 = Tacotron2Hybrid(num_chars=132, style_dim=128)
    vocoder = VocoderHybrid(style_dim=128)
    inference = HybridInference(tacotron2, vocoder, device=device)
    print("   ✓ Ready")
    
    # Test pipelines
    text = "ತಂತ್ರಜ್ಞಾನ"
    pipelines = ["standard", "advanced", "quality", "speed"]
    
    print(f"\n2. Synthesizing '{text}' with different pipelines...")
    
    os.makedirs("output/examples/pipelines", exist_ok=True)
    
    for pipeline in pipelines:
        print(f"   - Using {pipeline} pipeline...")
        audio = inference.synthesize(
            text,
            post_processing=pipeline
        )
        output_path = f"output/examples/pipelines/example_4_{pipeline}.wav"
        inference.save_audio(audio, output_path)
        print(f"     ✓ Saved: {output_path}")
    
    print("\n✓ All pipeline examples generated")


# ==============================================================================
# EXAMPLE 5: Batch Synthesis
# ==============================================================================

def example_5_batch_synthesis():
    """Example 5: Batch synthesis for multiple texts"""
    
    print("\n" + "="*80)
    print("EXAMPLE 5: Batch Synthesis")
    print("="*80)
    
    from src.hybrid.models import Tacotron2Hybrid, VocoderHybrid
    from src.hybrid.inference import HybridInference
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create inference
    print("\n1. Creating models and inference...")
    tacotron2 = Tacotron2Hybrid(num_chars=132, style_dim=128)
    vocoder = VocoderHybrid(style_dim=128)
    inference = HybridInference(tacotron2, vocoder, device=device)
    print("   ✓ Ready")
    
    # Batch texts
    texts = [
        "ನಮಸ್ಕಾರ",
        "ಕನ್ನಡ ಭಾಷೆ",
        "ತಂತ್ರಜ್ಞಾನ",
        "ಬುದ್ಧಿಮತ್ತೆ"
    ]
    
    print(f"\n2. Batch synthesizing {len(texts)} texts...")
    
    os.makedirs("output/examples/batch", exist_ok=True)
    
    audios = inference.synthesize_batch(texts, emotion="happy")
    
    for idx, (text, audio) in enumerate(zip(texts, audios), 1):
        if audio is not None:
            output_path = f"output/examples/batch/example_5_batch_{idx}.wav"
            inference.save_audio(audio, output_path)
            print(f"   ✓ [{idx}] {text} -> {output_path}")
    
    print("\n✓ Batch synthesis complete")


# ==============================================================================
# EXAMPLE 6: Using Unified Interface
# ==============================================================================

def example_6_unified_interface():
    """Example 6: Using unified interface for both approaches"""
    
    print("\n" + "="*80)
    print("EXAMPLE 6: Unified Interface - Both Approaches")
    print("="*80)
    
    from src.inference_unified import TTSInference
    from src.hybrid.models import Tacotron2Hybrid, VocoderHybrid
    from src.non_hybrid.models import Tacotron2Model, VocoderModel
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = "ಸಹಯೋಗ"
    
    os.makedirs("output/examples/unified", exist_ok=True)
    
    # Hybrid approach
    print("\n1. Using HYBRID approach...")
    tacotron2 = Tacotron2Hybrid(num_chars=132, style_dim=128)
    vocoder = VocoderHybrid(style_dim=128)
    hybrid_tts = TTSInference("hybrid", tacotron2, vocoder, device=device)
    
    audio_hybrid = hybrid_tts.synthesize(text, emotion="happy")
    hybrid_tts.save_audio(audio_hybrid, "output/examples/unified/example_6_hybrid.wav")
    print(f"   ✓ Info: {hybrid_tts.get_info()}")
    
    # Non-hybrid approach
    print("\n2. Using NON-HYBRID approach...")
    tacotron2 = Tacotron2Model(num_chars=132)
    vocoder = VocoderModel()
    non_hybrid_tts = TTSInference("non_hybrid", tacotron2, vocoder, device=device)
    
    audio_non_hybrid = non_hybrid_tts.synthesize(text)
    non_hybrid_tts.save_audio(audio_non_hybrid, "output/examples/unified/example_6_non_hybrid.wav")
    print(f"   ✓ Info: {non_hybrid_tts.get_info()}")
    
    print("\n✓ Both approaches synthesized successfully")


# ==============================================================================
# EXAMPLE 7: Advanced Prosody Control
# ==============================================================================

def example_7_prosody_control():
    """Example 7: Advanced prosody manipulation"""
    
    print("\n" + "="*80)
    print("EXAMPLE 7: Advanced Prosody Control")
    print("="*80)
    
    from src.hybrid.processors import ProsodyEnhancer
    import numpy as np
    
    # Create sample audio (sine wave)
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    print("\n1. Creating prosody enhancer...")
    enhancer = ProsodyEnhancer(sr)
    print("   ✓ Ready")
    
    os.makedirs("output/examples/prosody", exist_ok=True)
    
    print("\n2. Applying prosody effects...")
    
    # Original
    effects = {
        "original": audio,
        "pitch_up": enhancer.pitch_shift(audio, shift_factor=1.2),
        "pitch_down": enhancer.pitch_shift(audio, shift_factor=0.8),
        "faster": enhancer.time_stretch(audio, stretch_factor=0.8),
        "slower": enhancer.time_stretch(audio, stretch_factor=1.2),
        "compressed": enhancer.enhance_dynamics(audio, compression_ratio=4.0),
    }
    
    print("\n3. Saving effects...")
    for effect_name, audio_data in effects.items():
        # Simple save (would use soundfile in real code)
        print(f"   ✓ {effect_name}: {audio_data.shape}")
    
    print("\n✓ Prosody control examples complete")


# ==============================================================================
# EXAMPLE 8: Training Setup
# ==============================================================================

def example_8_training_setup():
    """Example 8: Setting up training for both approaches"""
    
    print("\n" + "="*80)
    print("EXAMPLE 8: Training Setup")
    print("="*80)
    
    from src.hybrid.models import Tacotron2Hybrid, VocoderHybrid
    from src.hybrid.training import HybridTrainer
    from src.non_hybrid.models import Tacotron2Model, VocoderModel
    from src.non_hybrid.training import StandardTrainer
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Hybrid training setup
    print("\n1. HYBRID Training Setup")
    print("   - Creating models...")
    tacotron2_h = Tacotron2Hybrid(num_chars=132, style_dim=128)
    vocoder_h = VocoderHybrid(style_dim=128)
    trainer_h = HybridTrainer(tacotron2_h, vocoder_h, device)
    print(f"   - Checkpoint dir: {trainer_h.checkpoint_dir}")
    print("   ✓ Setup complete")
    
    # Non-hybrid training setup
    print("\n2. NON-HYBRID Training Setup")
    print("   - Creating models...")
    tacotron2_nh = Tacotron2Model(num_chars=132)
    vocoder_nh = VocoderModel()
    trainer_nh = StandardTrainer(tacotron2_nh, vocoder_nh, device)
    print(f"   - Checkpoint dir: {trainer_nh.checkpoint_dir}")
    print("   ✓ Setup complete")
    
    print("\n3. Training configuration ready")
    print("   Note: Provide train_loader and val_loader to start training")
    print("   Example: trainer.train_epoch(train_loader, val_loader, epoch=0)")


# ==============================================================================
# EXAMPLE 9: Model Comparison
# ==============================================================================

def example_9_model_comparison():
    """Example 9: Compare model architectures"""
    
    print("\n" + "="*80)
    print("EXAMPLE 9: Model Architecture Comparison")
    print("="*80)
    
    from src.hybrid.models import Tacotron2Hybrid, VocoderHybrid
    from src.non_hybrid.models import Tacotron2Model, VocoderModel

    device = "cpu"  # Use CPU for quick comparison
    
    # Hybrid models
    print("\n1. HYBRID MODELS")
    print("-" * 40)
    
    tacotron2_h = Tacotron2Hybrid(num_chars=132, style_dim=128).to(device)
    vocoder_h = VocoderHybrid(style_dim=128).to(device)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    tac_params_h = count_parameters(tacotron2_h)
    voc_params_h = count_parameters(vocoder_h)
    
    print(f"Tacotron2Hybrid parameters: {tac_params_h:,}")
    print(f"VocoderHybrid parameters:   {voc_params_h:,}")
    print(f"Total Hybrid parameters:    {tac_params_h + voc_params_h:,}")
    
    # Non-hybrid models
    print("\n2. NON-HYBRID MODELS")
    print("-" * 40)
    
    tacotron2_nh = Tacotron2Model(num_chars=132).to(device)
    vocoder_nh = VocoderModel().to(device)
    
    tac_params_nh = count_parameters(tacotron2_nh)
    voc_params_nh = count_parameters(vocoder_nh)
    
    print(f"Tacotron2Model parameters:  {tac_params_nh:,}")
    print(f"VocoderModel parameters:    {voc_params_nh:,}")
    print(f"Total Non-Hybrid parameters: {tac_params_nh + voc_params_nh:,}")
    
    # Comparison
    print("\n3. COMPARISON")
    print("-" * 40)
    
    diff_percent = ((tac_params_h + voc_params_h) - (tac_params_nh + voc_params_nh)) / (tac_params_nh + voc_params_nh) * 100
    print(f"Hybrid has {diff_percent:.1f}% more parameters than Non-Hybrid")


# ==============================================================================
# Main: Run All Examples
# ==============================================================================

def main():
    """Run all examples"""
    
    examples = [
        ("1: Hybrid Basic", example_1_hybrid_basic),
        ("2: Non-Hybrid Basic", example_2_non_hybrid_basic),
        ("3: Hybrid Emotions", example_3_hybrid_emotions),
        ("4: Post-Processing Pipelines", example_4_hybrid_pipelines),
        ("5: Batch Synthesis", example_5_batch_synthesis),
        ("6: Unified Interface", example_6_unified_interface),
        ("7: Prosody Control", example_7_prosody_control),
        ("8: Training Setup", example_8_training_setup),
        ("9: Model Comparison", example_9_model_comparison),
    ]
    
    print("\n" + "="*80)
    print("KANNADA TTS - COMPLETE EXAMPLES")
    print("="*80)
    print("\nAvailable Examples:")
    for name, _ in examples:
        print(f"  - {name}")
    print("\nRun specific example: python examples.py <number>")
    print("Example: python examples.py 1")
    
    # Run all examples by default
    print("\n" + "="*80)
    print("Running ALL examples...")
    print("="*80)
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            logger.error(f"Error in {name}: {e}", exc_info=True)
    
    print("\n" + "="*80)
    print("✓ All examples completed!")
    print("Check output/examples/ for generated files")
    print("="*80 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example_num = int(sys.argv[1])
        examples = [
            example_1_hybrid_basic,
            example_2_non_hybrid_basic,
            example_3_hybrid_emotions,
            example_4_hybrid_pipelines,
            example_5_batch_synthesis,
            example_6_unified_interface,
            example_7_prosody_control,
            example_8_training_setup,
            example_9_model_comparison,
        ]
        if 1 <= example_num <= len(examples):
            examples[example_num - 1]()
        else:
            print(f"Invalid example number: {example_num}")
    else:
        main()
