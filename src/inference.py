#!/usr/bin/env python3
from TTS.api import TTS
import torch, os

os.makedirs("output/inference", exist_ok=True)

# Try custom model first, fallback to pretrained
model_dir = "output/tacotron2"
if os.path.exists(f"{model_dir}/best_model.pth"):
    tts = TTS(model_path=f"{model_dir}/best_model.pth", 
              config_path="config/tacotron2.json", 
              gpu=torch.cuda.is_available())
else:
    print("⚠️  No trained model found. Using pretrained English (for testing)")
    tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=False)

# Kannada test sentences
tests = [
    "ನಮಸ್ಕಾರ, ಇದು ಕನ್ನಡ TTS ಪರೀಕ್ಷೆ.",
    "ಬೆಂಗಳೂರು ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ.",
    "ನಮಗೆ ಒಳ್ಳೆಯ ದಿನ ಬಯಸುತ್ತೇನೆ."
]

for i, text in enumerate(tests):
    file_path = f"output/inference/test_{i+1}.wav"
    tts.tts_to_file(text=text, file_path=file_path)
    print(f"✅ Saved: {file_path}")

print("🎵 Check output/inference/ for audio files!")
