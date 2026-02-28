#!/usr/bin/env python3
import os, re, pandas as pd
import kagglehub

print("📥 Downloading Kannada-M dataset...")
path = kagglehub.dataset_download('skywalker290/kannada-m')
txt_dir = f"{path}/Kannada_M/txt"
wav_dir = f"{path}/Kannada_M/wav"

os.makedirs("data", exist_ok=True)

def clean_kannada_text(text):
    text = text.strip()
    text = re.sub(r"[^\u0C80-\u0CFF\s।,.!?]", "", text)
    return re.sub(r"\s+", " ", text).strip()

print("🔄 Creating metadata.csv...")
with open("data/metadata.csv", "w", encoding="utf-8") as f:
    count = 0
    for txt_file in os.listdir(txt_dir):
        if txt_file.endswith('.txt'):
            base = txt_file.replace('.txt', '')
            wav_path = f"{wav_dir}/{base}.wav"
            txt_path = f"{txt_dir}/{txt_file}"
            
            if os.path.exists(wav_path):
                with open(txt_path, 'r', encoding='utf-8') as t:
                    text = clean_kannada_text(t.read())
                f.write(f"{wav_path}|{text}\n")
                count += 1

print(f"✅ data/metadata.csv created: {count} samples")
