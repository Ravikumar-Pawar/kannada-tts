#!/usr/bin/env python3
"""
Kannada TTS Data Preparation Pipeline (Fixed)
Downloads Kannada-M dataset, validates pairs, cleans text, creates metadata
"""

import os
import re
import json
import librosa
import numpy as np
import pandas as pd
import kagglehub
from pathlib import Path
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("KANNADA TTS - DATA PREPARATION PIPELINE")
print("="*60)

# ============================================================================
# 1. DOWNLOAD DATASET
# ============================================================================
print("\n[1/6] Downloading Kannada-M dataset...")
try:
    path = kagglehub.dataset_download('skywalker290/kannada-m')
    txt_dir = f"{path}/Kannada_M/txt"
    wav_dir = f"{path}/Kannada_M/wav"
    print(f"Dataset: {path}")
except Exception as e:
    print(f"Error: {e}")
    exit(1)

os.makedirs("data", exist_ok=True)

# ============================================================================
# 2. TEXT CLEANING
# ============================================================================
def clean_kannada_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[^\u0C80-\u0CFF\s।,.!?;:-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ============================================================================
# 3. AUDIO VALIDATION
# ============================================================================
def validate_audio(wav_path: str) -> Tuple[bool, Dict]:
    try:
        if not os.path.exists(wav_path):
            return False, {"error": "File not found"}
        
        y, sr = librosa.load(wav_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        if duration < 1.0 or duration > 30.0:
            return False, {"error": f"Duration: {duration}s"}
        
        return True, {
            "duration": duration,
            "sr": sr,
            "peak": float(np.max(np.abs(y)))
        }
    except:
        return False, {"error": "Load failed"}

# ============================================================================
# 4. PROCESS FILES
# ============================================================================
print("\n[2/6] Processing files...")

metadata_list = []
failed_count = 0

for txt_file in sorted(os.listdir(txt_dir)):
    if not txt_file.endswith('.txt'):
        continue
        
    base = txt_file.replace('.txt', '')
    wav_path = f"{wav_dir}/{base}.wav"
    txt_path = f"{txt_dir}/{txt_file}"
    
    # Validate audio
    valid, audio_info = validate_audio(wav_path)
    if not valid:
        failed_count += 1
        continue
    
    # Clean text
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = clean_kannada_text(f.read())
        
        if len(text) < 3:
            failed_count += 1
            continue
            
        metadata_list.append({
            'wav_path': wav_path,
            'text': text,
            'duration': audio_info['duration']
        })
    except:
        failed_count += 1

print(f"Valid pairs: {len(metadata_list)}")
print(f"Failed: {failed_count}")

# ============================================================================
# 5. SAVE METADATA (FIXED)
# ============================================================================
print("\n[3/6] Saving metadata...")

df = pd.DataFrame(metadata_list)

# Ensure duration exists (FIX)
if 'duration' not in df.columns:
    print("Adding duration column...")
    df['duration'] = [librosa.get_duration(filename=row['wav_path']) 
                     for _, row in df.iterrows()]

# Sort by duration
df_sorted = df.sort_values('duration').reset_index(drop=True)

# Save LJSpeech format
with open("data/metadata.csv", "w", encoding="utf-8") as f:
    for _, row in df_sorted.iterrows():
        f.write(f"{row['wav_path']}|{row['text']}\n")

print(f"data/metadata.csv: {len(df_sorted)} samples")

# Save extended info
df_sorted.to_csv("data/metadata_extended.csv", index=False)

# ============================================================================
# 6. STATISTICS
# ============================================================================
print("\n[4/6] Statistics")

print(f"Duration - Min: {df_sorted['duration'].min():.2f}s, Max: {df_sorted['duration'].max():.2f}s, Mean: {df_sorted['duration'].mean():.2f}s")
print(f"Text length - Min: {df_sorted['text'].str.len().min()}, Max: {df_sorted['text'].str.len().max()}, Mean: {df_sorted['text'].str.len().mean():.1f}")

# Train/val/test split (85/7.5/7.5)
n_train = int(0.85 * len(df))
n_val = int(0.075 * len(df))

train_df = df_sorted[:n_train][['wav_path', 'text']]
val_df = df_sorted[n_train:n_train+n_val][['wav_path', 'text']]
test_df = df_sorted[n_train+n_val:][['wav_path', 'text']]

train_df.to_csv("data/train.csv", sep='|', header=False, index=False)
val_df.to_csv("data/val.csv", sep='|', header=False, index=False)
test_df.to_csv("data/test.csv", sep='|', header=False, index=False)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ============================================================================
# 7. SAVE DATASET INFO
# ============================================================================
dataset_info = {
    "total_samples": len(df),
    "train": len(train_df),
    "val": len(val_df),
    "test": len(test_df),
    "avg_duration": float(df['duration'].mean()),
    "avg_chars": float(df['text'].str.len().mean())
}

with open("data/dataset_info.json", "w") as f:
    json.dump(dataset_info, f, indent=2)

print("\n" + "="*60)
print("DATA PREPARATION COMPLETE")
print("="*60)
print("Files created:")
print("  data/metadata.csv")
print("  data/train.csv, data/val.csv, data/test.csv")
print("\nNext: python src/inference.py")
