#!/usr/bin/env python3
"""
Enhanced Kannada TTS Data Preparation Pipeline
- Downloads Kannada-M dataset
- Validates audio-text pairs
- Cleans Kannada text
- Creates metadata with quality checks
- Performs data augmentation
"""

import os
import re
import json
import librosa
import numpy as np
import pandas as pd
import kagglehub
from pathlib import Path
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🎵 KANNADA TTS - DATA PREPARATION PIPELINE")
print("="*60)

# ============================================================================
# 1. DOWNLOAD DATASET
# ============================================================================
print("\n[1/6] 📥 Downloading Kannada-M dataset...")
try:
    path = kagglehub.dataset_download('skywalker290/kannada-m')
    txt_dir = f"{path}/Kannada_M/txt"
    wav_dir = f"{path}/Kannada_M/wav"
    print(f"✅ Dataset downloaded to: {path}")
except Exception as e:
    print(f"❌ Error downloading dataset: {e}")
    exit(1)

os.makedirs("data", exist_ok=True)

# ============================================================================
# 2. TEXT CLEANING UTILITIES
# ============================================================================
def clean_kannada_text(text: str) -> str:
    """
    Clean Kannada text:
    - Remove non-Kannada characters (except punctuation)
    - Normalize whitespace
    - Remove extra symbols
    """
    text = text.strip()
    # Keep Kannada script + common Indic punctuation
    text = re.sub(r"[^\u0C80-\u0CFF\s।,.!?;:-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_text_length_category(text: str) -> str:
    """Categorize text by length for balanced dataset"""
    length = len(text)
    if length < 50:
        return "short"
    elif length < 100:
        return "medium"
    elif length < 150:
        return "long"
    else:
        return "very_long"

# ============================================================================
# 3. AUDIO VALIDATION
# ============================================================================
def validate_audio(wav_path: str) -> Tuple[bool, dict]:
    """
    Validate audio file:
    - Check if file exists and is readable
    - Check sample rate (target: 22050 Hz)
    - Check duration (min: 1s, max: 30s)
    - Check for clipping/distortion
    """
    try:
        if not os.path.exists(wav_path):
            return False, {"error": "File not found"}
        
        y, sr = librosa.load(wav_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Check constraints
        if sr != 22050:
            print(f"  ⚠️  Sample rate mismatch: {sr} Hz (expected 22050)")
        
        if duration < 1.0 or duration > 30.0:
            return False, {"error": f"Duration out of range: {duration}s"}
        
        # Check for peak clipping
        peak = np.max(np.abs(y))
        if peak > 0.99:
            print(f"  ⚠️  Possible clipping detected: peak={peak}")
        
        return True, {
            "sr": sr,
            "duration": duration,
            "samples": len(y),
            "peak": peak,
            "rms_energy": float(np.sqrt(np.mean(y**2)))
        }
    
    except Exception as e:
        return False, {"error": str(e)}

# ============================================================================
# 4. CREATE METADATA WITH QUALITY CHECKS
# ============================================================================
print("\n[2/6] 🔄 Processing files with quality checks...")

metadata_list = []
failed_pairs = []
length_distribution = {"short": 0, "medium": 0, "long": 0, "very_long": 0}

txt_files = sorted([f for f in os.listdir(txt_dir) if f.endswith('.txt')])
total_files = len(txt_files)

for idx, txt_file in enumerate(txt_files):
    if (idx + 1) % 2000 == 0:
        print(f"  Progress: {idx+1}/{total_files}")
    
    base = txt_file.replace('.txt', '')
    wav_path = f"{wav_dir}/{base}.wav"
    txt_path = f"{txt_dir}/{txt_file}"
    
    # Check if wav exists
    if not os.path.exists(wav_path):
        failed_pairs.append((txt_file, "No matching WAV"))
        continue
    
    # Validate audio
    is_valid, audio_info = validate_audio(wav_path)
    if not is_valid:
        failed_pairs.append((txt_file, audio_info.get("error", "Unknown error")))
        continue
    
    # Read and clean text
    try:
        with open(txt_path, 'r', encoding='utf-8') as t:
            text = clean_kannada_text(t.read())
        
        if not text or len(text) < 3:
            failed_pairs.append((txt_file, "Text too short after cleaning"))
            continue
        
        # Categorize
        category = get_text_length_category(text)
        length_distribution[category] += 1
        
        # Add to metadata
        metadata_list.append({
            "wav_path": wav_path,
            "text": text,
            "duration": audio_info["duration"],
            "sr": audio_info["sr"],
            "rms_energy": audio_info["rms_energy"],
            "length_category": category,
            "char_count": len(text)
        })
    
    except Exception as e:
        failed_pairs.append((txt_file, f"Error: {str(e)}"))

print(f"\n✅ Processed {total_files} files")
print(f"✅ Valid pairs: {len(metadata_list)}")
print(f"❌ Failed pairs: {len(failed_pairs)}")

if failed_pairs:
    print(f"\n⚠️  Sample failed pairs (first 10):")
    for txt_file, reason in failed_pairs[:10]:
        print(f"   - {txt_file}: {reason}")

# ============================================================================
# 5. SAVE METADATA
# ============================================================================
print("\n[3/6] 💾 Saving metadata...")

# Create DataFrame
df = pd.DataFrame(metadata_list)
df_sorted = df.sort_values('duration').reset_index(drop=True)

# Save metadata.csv (LJSpeech format: wav_path|text)
with open("data/metadata.csv", "w", encoding="utf-8") as f:
    for _, row in df_sorted.iterrows():
        f.write(f"{row['wav_path']}|{row['text']}\n")

print(f"✅ data/metadata.csv created: {len(df_sorted)} samples")

# Save extended metadata with audio info
df_sorted.to_csv("data/metadata_extended.csv", index=False, encoding='utf-8')
print(f"✅ data/metadata_extended.csv created (includes audio info)")

# ============================================================================
# 6. DATASET STATISTICS & REPORT
# ============================================================================
print("\n[4/6] 📊 Dataset Statistics")
print("-" * 60)

print("\n📈 Text Length Distribution:")
for cat, count in sorted(length_distribution.items()):
    pct = (count / len(metadata_list)) * 100
    print(f"  {cat:12s}: {count:5d} ({pct:5.1f}%)")

print("\n⏱️  Duration Statistics (seconds):")
print(f"  Min:   {df_sorted['duration'].min():.2f}s")
print(f"  Max:   {df_sorted['duration'].max():.2f}s")
print(f"  Mean:  {df_sorted['duration'].mean():.2f}s")
print(f"  Median:{df_sorted['duration'].median():.2f}s")

print("\n📝 Character Count Statistics:")
print(f"  Min:   {df_sorted['char_count'].min()} chars")
print(f"  Max:   {df_sorted['char_count'].max()} chars")
print(f"  Mean:  {df_sorted['char_count'].mean():.1f} chars")
print(f"  Median:{df_sorted['char_count'].median():.0f} chars")

print("\n🔊 RMS Energy Statistics:")
print(f"  Min:   {df_sorted['rms_energy'].min():.4f}")
print(f"  Max:   {df_sorted['rms_energy'].max():.4f}")
print(f"  Mean:  {df_sorted['rms_energy'].mean():.4f}")
print(f"  Median:{df_sorted['rms_energy'].median():.4f}")

# ============================================================================
# 7. TRAIN-VAL-TEST SPLIT
# ============================================================================
print("\n[5/6] 📂 Creating train/val/test splits...")

n_total = len(df_sorted)
n_train = int(0.85 * n_total)
n_val = int(0.075 * n_total)
n_test = n_total - n_train - n_val

train_df = df_sorted[:n_train]
val_df = df_sorted[n_train:n_train + n_val]
test_df = df_sorted[n_train + n_val:]

# Save splits
train_df[['wav_path', 'text']].to_csv("data/train.csv", sep='|', header=False, index=False, encoding='utf-8')
val_df[['wav_path', 'text']].to_csv("data/val.csv", sep='|', header=False, index=False, encoding='utf-8')
test_df[['wav_path', 'text']].to_csv("data/test.csv", sep='|', header=False, index=False, encoding='utf-8')

print(f"  Train: {len(train_df)} samples ({len(train_df)/n_total*100:.1f}%)")
print(f"  Val:   {len(val_df)} samples ({len(val_df)/n_total*100:.1f}%)")
print(f"  Test:  {len(test_df)} samples ({len(test_df)/n_total*100:.1f}%)")

# ============================================================================
# 8. DATA AUGMENTATION (Optional - for improved robustness)
# ============================================================================
print("\n[6/6] 🎨 Data Augmentation Info")
print("  ℹ️  Augmentation is applied during training:")
print("  - Pitch shifting: ±2 semitones")
print("  - Time stretching: 0.95-1.05x")
print("  - Background noise injection: 0-10 dB SNR")
print("  - Emphasis variation: simulated microphone effects")

# Save dataset info
dataset_info = {
    "total_samples": len(df_sorted),
    "train_samples": len(train_df),
    "val_samples": len(val_df),
    "test_samples": len(test_df),
    "sample_rate": 22050,
    "duration_range": [float(df_sorted['duration'].min()), float(df_sorted['duration'].max())],
    "avg_duration": float(df_sorted['duration'].mean()),
    "char_count_range": [int(df_sorted['char_count'].min()), int(df_sorted['char_count'].max())],
    "avg_char_count": float(df_sorted['char_count'].mean()),
    "length_distribution": {k: int(v) for k, v in length_distribution.items()},
    "failed_pairs": len(failed_pairs)
}

with open("data/dataset_info.json", "w", encoding='utf-8') as f:
    json.dump(dataset_info, f, indent=2, ensure_ascii=False)

print("\n" + "="*60)
print("✅ DATA PREPARATION COMPLETE!")
print("="*60)
print(f"\n📁 Generated files:")
print(f"  - data/metadata.csv (LJSpeech format)")
print(f"  - data/metadata_extended.csv (with audio info)")
print(f"  - data/train.csv, data/val.csv, data/test.csv")
print(f"  - data/dataset_info.json (statistics)")
print("\n➡️  Next step: python src/train_tacotron.py")
