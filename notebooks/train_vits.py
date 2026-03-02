"""Colab-friendly training script for VITS Hybrid Model.

Usage (Colab):
  pip install -r requirements.txt
  python notebooks/train_vits.py --epochs 1 --batch-size 2 --dummy

This script is intentionally minimal: it demonstrates instantiation of the
VITS and Hybrid trainer from the repo, runs a small dummy training step when
no dataset is provided, and saves a checkpoint into `models/` for later use.
"""
import argparse
import os
import torch
from src.hybrid.models.vits_model import VITS
from src.hybrid.models.vocoder_hybrid import VocoderHybrid
from src.hybrid.training import HybridTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--dummy', action='store_true', help='Run a dummy training step with random tensors')
    p.add_argument('--out-dir', type=str, default='models')
    return p.parse_args()


def make_dummy_batch(batch_size, seq_len=32, mel_len=160, device='cpu'):
    import torch
    text = torch.randint(1, 132, (batch_size, seq_len), dtype=torch.long, device=device)
    text_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
    mels = torch.randn(batch_size, mel_len, 80, device=device)
    gates = torch.zeros(batch_size, mel_len, device=device)
    audio = torch.randn(batch_size, 1, mel_len * 256, device=device)
    return {
        'text': text,
        'text_lengths': text_lengths,
        'mels': mels,
        'gates': gates,
        'audio': audio,
    }


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs('output/hybrid_checkpoints', exist_ok=True)

    # Instantiate models
    vits = VITS()
    vocoder = VocoderHybrid()

    trainer = HybridTrainer(vits, vocoder, device=device)

    if args.dummy:
        print('Running dummy training step...')
        batch = make_dummy_batch(args.batch_size, device=device)
        # emulate a single epoch by calling train_tacotron2_step and train_vocoder_step
        tac_losses = trainer.train_tacotron2_step(
            batch['text'], batch['text_lengths'], batch['mels'], batch['gates'], reference_mels=None
        )
        voc_losses = trainer.train_vocoder_step(batch['mels'], batch['audio'], style_embedding=None)
        print('Tacotron2 losses:', tac_losses)
        print('Vocoder losses:', voc_losses)
        # Save checkpoints
        vits_path = os.path.join(args.out_dir, 'vits_kannada.pth')
        voc_path = os.path.join(args.out_dir, 'vocoder_kannada.pth')
        vits.save_model(vits_path)
        vocoder.save_model(voc_path)
        trainer.save_checkpoint(epoch=1, metrics={'dummy': True})
        print(f'Models saved: {vits_path}, {voc_path}')
    else:
        print('No dataset configured. Use --dummy for a small test run or extend this script to')
        print('provide a proper DataLoader.')


if __name__ == '__main__':
    main()
