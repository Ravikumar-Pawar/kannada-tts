"""Colab-friendly inference script for VITS Hybrid Model.

Usage (Colab):
  pip install -r requirements.txt
  python notebooks/generate_vits.py --vits models/vits_kannada.pth --vocoder models/vocoder_kannada.pth --text "ನಮಸ್ಕಾರ"

The script will load the VITS and Vocoder checkpoints, run inference for a short text,
and write `output/generated.wav` which you can download from Colab.
"""
import argparse
import os
import torch
import soundfile as sf
from src.hybrid.models.vits_model import VITS
from src.hybrid.models.vocoder_hybrid import VocoderHybrid


def simple_tokenize(text: str, max_len: int = 256):
    # Very small placeholder tokenizer: convert chars to small integer ids
    ids = [ord(c) % 132 for c in text][:max_len]
    return ids


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--vits', type=str, required=True)
    p.add_argument('--vocoder', type=str, required=True)
    p.add_argument('--text', type=str, default='ನಮಸ್ಕಾರ')
    p.add_argument('--out', type=str, default='output/generated.wav')
    p.add_argument('--sr', type=int, default=22050)
    return p.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vits = VITS()
    vocoder = VocoderHybrid()

    if os.path.exists(args.vits):
        vits.load_model(args.vits)
    else:
        print('VITS checkpoint not found:', args.vits)
    if os.path.exists(args.vocoder):
        vocoder.load_model(args.vocoder)
    else:
        print('Vocoder checkpoint not found:', args.vocoder)

    vits.to(device)
    vocoder.to(device)
    vits.eval()
    vocoder.eval()

    ids = simple_tokenize(args.text)
    import torch
    text_tensor = torch.LongTensor([ids]).to(device)
    text_lengths = torch.LongTensor([len(ids)]).to(device)

    with torch.no_grad():
        out = vits(text_tensor, text_lengths, mels=None)
        # VITS returns mel in 'mel_output'
        mel = out.get('mel_output')
        if mel is None:
            print('VITS did not produce mel output')
            return
        # mel shape: (1, mel_length, mel_channels)
        mel = mel.cpu()
        # Vocoder expects mel (batch, mel_steps, 80)
        audio = vocoder.infer(mel, style=None, vocoder_type='fast')
        # audio shape: (batch, 1, samples)
        audio_np = audio.squeeze(0).squeeze(0).cpu().numpy()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sf.write(args.out, audio_np, samplerate=args.sr)
    print('Generated audio saved to', args.out)


if __name__ == '__main__':
    main()
