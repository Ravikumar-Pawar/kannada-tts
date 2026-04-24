# Dataset.py - Unified Dataset Manager

## Overview

`dataset.py` is a single consolidated script for all dataset operations:
- ✅ **Download** from OpenSLR
- ✅ **Prepare** (extract & organize)
- ✅ **Organize** custom datasets
- ✅ **Analyze** datasets
- ✅ **Full pipeline** (download + prepare + analyze)

Replaces: `prepare_dataset.py`, `organize_dataset.py`, `analyze_dataset.py`

---

## Quick Commands

### 1. Download from OpenSLR
```bash
python dataset.py download
```
Downloads Kannada dataset from OpenSLR (1000+ samples, ~500MB)

### 2. Extract & Prepare
```bash
python dataset.py prepare
```
Extracts WAV files and creates text files from metadata

### 3. Complete Pipeline (Recommended)
```bash
python dataset.py full
```
Does everything: download → extract → prepare → analyze

### 4. Organize Custom Dataset
```bash
python dataset.py organize \
    --data-dir data/kannada_tts_dataset \
    --audio-source /path/to/audio \
    --text-source /path/to/text \
    --mapping
```
Organizes your own Kannada audio+text files

### 5. Analyze Dataset
```bash
python dataset.py analyze \
    --data-dir data/kannada_tts_dataset \
    --save-report
```
Validates and analyzes dataset with detailed statistics

---

## Detailed Usage

### Download
```bash
# Download from OpenSLR
python dataset.py download

# Force re-download
python dataset.py download --force
```

**Output**: 
- `data/kannada_tts_dataset/dataset.zip` (WAV files, ~500MB)
- `data/kannada_tts_dataset/metadata.tsv` (text metadata)

---

### Prepare
```bash
# Extract and prepare
python dataset.py prepare

# Force re-processing
python dataset.py prepare --force
```

**Output**:
- `data/kannada_tts_dataset/wav/` (1000+ audio files @ 22050Hz)
- `data/kannada_tts_dataset/txt/` (1000+ text files)

---

### Full Pipeline
```bash
# Complete download + prepare + analyze
python dataset.py full

# Skip final analysis
python dataset.py full --skip-analyze

# Force re-processing
python dataset.py full --force
```

**Time**: 30-60 minutes (download) + 5-10 minutes (extract/prepare)

---

### Organize Custom Data
```bash
# Organize audio only
python dataset.py organize \
    --data-dir data/kannada_tts_dataset \
    --audio-source /path/to/audio

# Organize text only
python dataset.py organize \
    --data-dir data/kannada_tts_dataset \
    --text-source /path/to/text

# Organize both with mapping
python dataset.py organize \
    --data-dir data/kannada_tts_dataset \
    --audio-source /path/to/audio \
    --text-source /path/to/text \
    --mapping

# Custom sample rate
python dataset.py organize \
    --data-dir data/kannada_tts_dataset \
    --audio-source /path/to/audio \
    --sample-rate 16000
```

**Features**:
- Auto-resamples audio to 22050Hz (default)
- Matches audio-text file pairs
- UTF-8 encoding verification
- Creates mapping file (optional)

---

### Analyze
```bash
# Quick analysis
python dataset.py analyze --data-dir data/kannada_tts_dataset

# Save detailed report
python dataset.py analyze \
    --data-dir data/kannada_tts_dataset \
    --save-report
```

**Output**:
- Console statistics (files, sizes, duration, sample rates, etc.)
- Optional: `analysis_report.txt` file

**Analyzes**:
- File matching rates
- Audio properties (duration, sample rate, size)
- Text properties (word count, Kannada characters)
- Sample data preview
- Issue detection

---

## Workflow Examples

### Example 1: OpenSLR (Recommended for quick start)
```bash
# One command - everything!
python dataset.py full

# Then train
python training/train_hkl_vits.py \
    --config configs/hkl_vits_config.json \
    --data_dir data/kannada_tts_dataset
```

### Example 2: Custom Kannada Data
```bash
# Organize your data
python dataset.py organize \
    --data-dir data/kannada_tts_dataset \
    --audio-source /my/kannada/audio \
    --text-source /my/kannada/text \
    --mapping

# Verify
python dataset.py analyze \
    --data-dir data/kannada_tts_dataset \
    --save-report

# Train
python training/train_hkl_vits.py \
    --config configs/hkl_vits_config.json \
    --data_dir data/kannada_tts_dataset
```

### Example 3: Combine Multiple Datasets
```bash
# Step 1: Download OpenSLR
python dataset.py full --skip-analyze

# Step 2: Add more data
python dataset.py organize \
    --data-dir data/kannada_tts_dataset \
    --audio-source /additional/audio \
    --text-source /additional/text

# Step 3: Verify combined dataset
python dataset.py analyze \
    --data-dir data/kannada_tts_dataset \
    --save-report
```

---

## Input Requirements

### For Organize Command
- **Audio files**: .wav or .mp3 (any sample rate)
- **Text files**: .txt (any encoding)
- **Naming**: Files must match by stem
  - `sample_001.wav` ↔ `sample_001.txt`
  - `kn_m_00001.wav` ↔ `kn_m_00001.txt`

### Expected Output Structure
```
data/kannada_tts_dataset/
├── wav/           # Audio (22050 Hz)
│   ├── file_001.wav
│   ├── file_002.wav
│   └── ...
├── txt/           # Text (UTF-8)
│   ├── file_001.txt
│   ├── file_002.txt
│   └── ...
└── metadata.tsv   # (if downloaded from OpenSLR)
```

---

## Options Summary

```
download            Download from OpenSLR
  --force          Force re-download

prepare             Extract and prepare
  --force          Force re-processing

organize            Organize custom data
  --data-dir       Dataset directory (required)
  --audio-source   Audio source directory
  --text-source    Text source directory
  --sample-rate    Target sample rate (default 22050)
  --mapping        Generate mapping file

analyze             Analyze dataset
  --data-dir       Dataset directory (required)
  --save-report    Save report to file

full                Complete pipeline
  --force          Force re-processing
  --skip-analyze   Skip final analysis
```

---

## Examples in One Place

```bash
# 1. Full auto pipeline
python dataset.py full

# 2. Organize custom data
python dataset.py organize --data-dir data/kannada_tts_dataset --audio-source /audio --text-source /text --mapping

# 3. Analyze with report
python dataset.py analyze --data-dir data/kannada_tts_dataset --save-report

# 4. Download only
python dataset.py download

# 5. Prepare only
python dataset.py prepare

# 6. Re-organize with force
python dataset.py organize --data-dir data/kannada_tts_dataset --audio-source /audio --force 2>/dev/null || echo "organize doesn't have --force, use organize command directly"
```

---

## Status Check

After any operation, verify with:
```bash
python dataset.py analyze --data-dir data/kannada_tts_dataset --save-report
```

Look for:
- ✓ "Matching rate" near 100%
- ✓ Sample rates showing 22050 Hz
- ✓ Duration statistics reasonable (2-30s)
- ✓ "No issues detected!"

---

## Benefits of Consolidated Script

✅ **Single file** - Easier to manage  
✅ **Unified interface** - Consistent subcommands  
✅ **No file conflicts** - One source of truth  
✅ **Modular classes** - Reusable in Python code  
✅ **Better organization** - Clear subcommand structure  
✅ **Same functionality** - All three tools in one  

---

## Migration from Old Scripts

| Old Script | New Command |
|-----------|------------|
| `python prepare_dataset.py` | `python dataset.py full` |
| `python organize_dataset.py --data-dir ... --audio-source ...` | `python dataset.py organize --data-dir ... --audio-source ...` |
| `python analyze_dataset.py --data-dir ...` | `python dataset.py analyze --data-dir ...` |

---

## Next Steps

1. Run: `python dataset.py full`
2. Wait for completion
3. Verify: `python dataset.py analyze --data-dir data/kannada_tts_dataset --save-report`
4. Train: `python training/train_hkl_vits.py --config configs/hkl_vits_config.json --data_dir data/kannada_tts_dataset`

---

**Version**: 1.0.0  
**Date**: April 2026  
**Status**: ✅ Production Ready
