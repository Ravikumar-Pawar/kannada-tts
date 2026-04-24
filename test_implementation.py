#!/usr/bin/env python3
"""
Validation script to test HKL-VITS implementation
Verifies all components are correctly implemented and integrated
"""

import torch
import torch.nn as nn
import sys
import json
from pathlib import Path


def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from hkl_vits.grapheme_encoder import GraphemeEncoder
        from hkl_vits.phoneme_encoder import PhonemeEncoder
        from hkl_vits.fusion_layer import FusionLayer
        from hkl_vits.prosody_encoder import ProsodyEncoder
        from hkl_vits.kannada_g2p import KannadaG2P
        from hkl_vits.dataset_loader import KannadaTTSDataset, get_dataloaders
        from hkl_vits.loss_functions import HKLVITSLoss, DiscriminatorLoss, GeneratorLoss
        from hkl_vits.hkl_vits_model import HKLVITS
        from hkl_vits.inference import HKLVITSInference
        from training.train_hkl_vits import HKLVITSTrainer
        from training.evaluate import HKLVITSEvaluator
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_g2p_converter():
    """Test Kannada G2P converter"""
    print("\nTesting Kannada G2P Converter...")
    try:
        from hkl_vits.kannada_g2p import KannadaG2P
        
        g2p = KannadaG2P()
        
        # Test 1: Random text conversion
        test_text = "ನಮಸ್ತೆ"  # "Namaste" in Kannada
        phonemes = g2p.grapheme_to_phoneme(test_text)
        print(f"  Text: {test_text}")
        print(f"  Phonemes: {phonemes}")
        assert isinstance(phonemes, list), "Phonemes should be a list"
        assert len(phonemes) > 0, "Should produce phonemes"
        
        # Test 2: Batch conversion
        texts = ["ನಮಸ್ತೆ", "ಕನ್ನಡ"]
        ids, lengths = g2p.batch_text_to_phoneme_ids(texts)
        print(f"  Batch shape: {ids.shape}, lengths: {lengths}")
        assert ids.shape[0] == len(texts), "Batch size should match"
        
        print("✓ G2P converter tests passed")
        return True
    except Exception as e:
        print(f"✗ G2P converter test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_encoders():
    """Test encoder modules"""
    print("\nTesting Encoder Modules...")
    try:
        from hkl_vits.grapheme_encoder import GraphemeEncoder
        from hkl_vits.phoneme_encoder import PhonemeEncoder
        from hkl_vits.fusion_layer import FusionLayer
        from hkl_vits.prosody_encoder import ProsodyEncoder
        
        batch_size = 2
        seq_len = 10
        hidden_dim = 256
        
        # Test grapheme encoder
        grapheme_encoder = GraphemeEncoder(vocab_size=150, hidden_dim=hidden_dim)
        grapheme_input = torch.randint(0, 150, (batch_size, seq_len))
        grapheme_output = grapheme_encoder(grapheme_input)
        assert grapheme_output.shape == (batch_size, seq_len, hidden_dim), f"Grapheme output shape mismatch: {grapheme_output.shape}"
        print(f"  Grapheme encoder output shape: {grapheme_output.shape} ✓")
        
        # Test phoneme encoder
        phoneme_encoder = PhonemeEncoder(phoneme_vocab=80, hidden_dim=hidden_dim)
        phoneme_input = torch.randint(0, 80, (batch_size, seq_len))
        phoneme_output = phoneme_encoder(phoneme_input)
        assert phoneme_output.shape == (batch_size, seq_len, hidden_dim), f"Phoneme output shape mismatch: {phoneme_output.shape}"
        print(f"  Phoneme encoder output shape: {phoneme_output.shape} ✓")
        
        # Test fusion layer
        fusion_layer = FusionLayer(hidden_dim=hidden_dim)
        fused_output = fusion_layer(grapheme_output, phoneme_output)
        assert fused_output.shape == (batch_size, seq_len, hidden_dim), f"Fused output shape mismatch: {fused_output.shape}"
        print(f"  Fusion layer output shape: {fused_output.shape} ✓")
        
        # Test prosody encoder
        prosody_encoder = ProsodyEncoder(hidden_dim=hidden_dim)
        pitch = torch.randn(batch_size, seq_len, 1)
        energy = torch.randn(batch_size, seq_len, 1)
        prosody_output = prosody_encoder(pitch, energy)
        assert prosody_output.shape == (batch_size, seq_len, hidden_dim), f"Prosody output shape mismatch: {prosody_output.shape}"
        print(f"  Prosody encoder output shape: {prosody_output.shape} ✓")
        
        print("✓ All encoder tests passed")
        return True
    except Exception as e:
        print(f"✗ Encoder test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_hklvits_model():
    """Test main HKL-VITS model"""
    print("\nTesting HKL-VITS Model...")
    try:
        from hkl_vits.hkl_vits_model import HKLVITS
        
        # Create model with small dimensions for testing
        model = HKLVITS(
            vocab_size=150,
            phoneme_vocab=80,
            n_mels=80,
            hidden_dim=128,
            num_layers=2,
            num_flows=2
        )
        
        # Create dummy inputs
        batch_size = 2
        seq_len = 10
        mel_len = 30
        
        text = torch.randint(0, 150, (batch_size, seq_len))
        phonemes = torch.randint(0, 80, (batch_size, seq_len))
        pitch = torch.randn(batch_size, seq_len, 1)
        energy = torch.randn(batch_size, seq_len, 1)
        mel_target = torch.randn(batch_size, 80, mel_len)
        
        # Forward pass
        model.train()
        outputs = model(
            text=text,
            phonemes=phonemes,
            pitch=pitch,
            energy=energy,
            mel_target=mel_target
        )
        
        assert 'waveform' in outputs, "Output should contain waveform"
        assert 'z' in outputs, "Output should contain latent space"
        print(f"  Waveform shape: {outputs['waveform'].shape} ✓")
        print(f"  Latent shape: {outputs['z'].shape} ✓")
        
        print("✓ HKL-VITS model test passed")
        return True
    except Exception as e:
        print(f"✗ HKL-VITS model test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_functions():
    """Test loss functions"""
    print("\nTesting Loss Functions...")
    try:
        from hkl_vits.loss_functions import HKLVITSLoss, DiscriminatorLoss, GeneratorLoss
        
        batch_size = 2
        mel_len = 30
        
        # Create loss functions
        criterion = HKLVITSLoss()
        disc_loss = DiscriminatorLoss()
        gen_loss = GeneratorLoss()
        
        # Create dummy data
        mel_pred = torch.randn(batch_size, 80, mel_len)
        mel_target = torch.randn(batch_size, 80, mel_len)
        pitch_pred = torch.randn(batch_size, mel_len)
        pitch_target = torch.randn(batch_size, mel_len)
        energy_pred = torch.randn(batch_size, mel_len)
        energy_target = torch.randn(batch_size, mel_len)
        mu = torch.randn(batch_size, 128, mel_len)
        log_var = torch.randn(batch_size, 128, mel_len)
        
        # Compute losses
        total_loss, loss_dict = criterion(
            mel_pred=mel_pred,
            mel_target=mel_target,
            pitch_pred=pitch_pred,
            pitch_target=pitch_target,
            energy_pred=energy_pred,
            energy_target=energy_target,
            mu=mu,
            log_var=log_var
        )
        
        print(f"  Total loss: {total_loss.item():.4f} ✓")
        print(f"  Loss components: {list(loss_dict.keys())} ✓")
        
        # Test discriminator loss
        real_pred = torch.randn(batch_size, 1)
        fake_pred = torch.randn(batch_size, 1)
        d_loss = disc_loss(real_pred, fake_pred)
        print(f"  Discriminator loss: {d_loss.item():.4f} ✓")
        
        # Test generator loss
        g_loss = gen_loss(fake_pred)
        print(f"  Generator loss: {g_loss.item():.4f} ✓")
        
        print("✓ Loss function tests passed")
        return True
    except Exception as e:
        print(f"✗ Loss function test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration():
    """Test configuration loading"""
    print("\nTesting Configuration...")
    try:
        config_path = Path('configs/hkl_vits_config.json')
        
        if not config_path.exists():
            print(f"✗ Config file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check required keys
        required_keys = ['model', 'training', 'loss_weights', 'data', 'logging']
        for key in required_keys:
            assert key in config, f"Missing config key: {key}"
        
        print(f"  Config keys: {list(config.keys())} ✓")
        print(f"  Model hidden dim: {config['model']['hidden_dim']} ✓")
        print(f"  Training batch size: {config['training']['batch_size']} ✓")
        
        print("✓ Configuration test passed")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("HKL-VITS Implementation Validation")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("G2P Converter", test_g2p_converter),
        ("Encoder Modules", test_encoders),
        ("Loss Functions", test_loss_functions),
        ("HKL-VITS Model", test_hklvits_model),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {str(e)}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:30s} {status}")
    
    print("="*60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Implementation is ready for training.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please fix the issues above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
