# Dataset Preparation Guide for HKL-VITS

This guide explains how to prepare datasets for training the Kannada HKL-VITS model.

## 📋 Table of Contents

1. [Quick Start (Auto Dataset)](#quick-start-auto-dataset)
2. [Manual Dataset Organization](#manual-dataset-organization)
3. [Dataset Structure](#dataset-structure)
4. [Data Validation](#data-validation)
5. [Troubleshooting](#troubleshooting)

---

## Quick Start (Auto Dataset)

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt
```

Required packages:
- `requests` - for downloading
- `pandas` - for metadata processing
- `tqdm` - for progress bars
- `librosa` - for audio processing
- `soundfile` - for audio I/O

### Option 1: Full Automatic Pipeline

Download, extract, and prepare the OpenSLR Kannada dataset in one command:

```bash
python prepare_dataset.py
```

This will:
1. ✅ Download audio (WAV zip) from OpenSLR
2. ✅ Download metadata (TSV) from OpenSLR
3. ✅ Extract all audio files
4. ✅ Generate text files from metadata
5. ✅ Validate the dataset

**Expected duration**: 30-60 minutes (depending on internet speed)

**Output location**: `data/kannada_tts_dataset/`

### Option 2: Skip Download (if already downloaded)

```bash
python prepare_dataset.py --no-download
```

### Option 3: Force Re-process Everything

```bash
python prepare_dataset.py --force
```

### Option 4: Only Validate Existing Dataset

```bash
python prepare_dataset.py --no-download --no-extract --no-prepare
```

---

## Manual Dataset Organization

If you have your own Kannada speech dataset, use the organization helper:

### Step 1: Prepare Your Data

Organize your files into two directories:

```
my_kannada_data/
├── audio/
│   ├── speaker_001.wav
│   ├── speaker_002.wav
│   └── ...
└── text/
    ├── speaker_001.txt
    ├── speaker_002.txt
    └── ...
```

### Step 2: Run Organization Script

```bash
python organize_dataset.py \
    --data-dir data/kannada_tts_dataset \
    --audio-source /path/to/my_kannada_data/audio \
    --text-source /path/to/my_kannada_data/text \
    --sample-rate 22050
```

### Step 3: Optional - Generate Mapping

```bash
python organize_dataset.py \
    --data-dir data/kannada_tts_dataset \
    --audio-source /path/to/my_kannada_data/audio \
    --text-source /path/to/my_kannada_data/text \
    --mapping
```

This creates `dataset_mapping.txt` showing which files are matched.

---

## Dataset Structure

After preparation, your dataset will be organized as:

```
data/
└── kannada_tts_dataset/          # Main dataset directory
    ├── wav/                      # Audio files (all .wav)
    │   ├── sample_001.wav
    │   ├── sample_002.wav
    │   ├── sample_003.wav
    │   └── ...
    ├── txt/                      # Text transcriptions (all .txt)
    │   ├── sample_001.txt        # "ನಮಸ್ತೆ"
    │   ├── sample_002.txt        # "ಧನ್ಯವಾದ"
    │   ├── sample_003.txt        # "ಈ ಸುಂದರ ದಿನ"
    │   └── ...
    ├── metadata.tsv              # Metadata file (optional, for reference)
    └── dataset_mapping.txt       # Mapping file (if generated)
```

### Important:
- ✅ **All filenames must match** (except extensions)
  - `sample_001.wav` → `sample_001.txt`
  - `kn_m_00001.wav` → `kn_m_00001.txt`
- ✅ **All audio must be 22050 Hz** (standardized)
- ✅ **All text must be UTF-8 encoded** (Kannada characters)

---

## Audio Requirements

| Specification | Requirement |
|--------------|-------------|
| **Format** | WAV |
| **Sample Rate** | 22050 Hz |
| **Bit Depth** | 16-bit |
| **Mono/Stereo** | Any (will be converted to mono) |
| **Duration** | 2-15 seconds recommended |
| **Total Size** | 100+ samples minimum |

### Audio Quality Guidelines

**Good samples:**
- ✅ Clear, natural speech
- ✅ Minimal background noise
- ✅ Normal speaking speed
- ✅ Neutral emotion

**Avoid:**
- ❌ Heavily processed audio
- ❌ Very high/low speed speech
- ❌ Strong emotions/prosody
- ❌ Music or sound effects

---

## Text Requirements

### Kannada Script

All text must be in **Kannada script** (ಕನ್ನಡ):

**Good examples:**
```
ನಮಸ್ತೆ
ಧನ್ಯವಾದ ಈ ಸುಂದರ ದಿನ
ನಮ್ಮ ನೆಡೆಯಲ್ಲಿ ಸುಂದರ ಮಳೆ ಬರುತ್ತಿದೆ
```

**Avoid:**
```
namaste           ❌ (English transliteration)
ನಮಸ್ತೆ।।।     ❌ (Extra punctuation)
नमस्ते         ❌ (Hindi script)
```

### Text Guidelines

- Length: 5-80 words per sample
- Should match pronunciation in audio
- Remove extra punctuation (keep minimal)
- One line per file

---

## Data Validation

### Automatic Validation

The preparation script automatically checks:
- ✅ All WAV files exist in proper format
- ✅ All TXT files exist and are UTF-8 encoded
- ✅ Matching pairs (wav ↔ txt)
- ✅ Audio file sizes and counts

### Manual Validation

To manually check your dataset:

```bash
python prepare_dataset.py --no-download --no-extract --no-prepare
```

This will display:
- Number of WAV files
- Number of TXT files
- Matched pairs
- Total audio size
- Sample data preview

### Check File Matching

```python
from pathlib import Path

dataset_dir = Path("data/kannada_tts_dataset")
wav_dir = dataset_dir / "wav"
txt_dir = dataset_dir / "txt"

wav_files = {f.stem for f in wav_dir.glob("*.wav")}
txt_files = {f.stem for f in txt_dir.glob("*.txt")}

unmatched_wav = wav_files - txt_files
unmatched_txt = txt_files - wav_files

print(f"Matched pairs: {len(wav_files & txt_files)}")
if unmatched_wav:
    print(f"WAV without TXT: {unmatched_wav}")
if unmatched_txt:
    print(f"TXT without WAV: {unmatched_txt}")
```

---

## Using Prepared Dataset with Training

After dataset preparation, train the model:

```bash
python training/train_hkl_vits.py \
    --config configs/hkl_vits_config.json \
    --data_dir data/kannada_tts_dataset \
    --gpu 0
```

The dataset loader (`hkl_vits/dataset_loader.py`) will:
1. Load audio from `data/kannada_tts_dataset/wav/`
2. Load text from `data/kannada_tts_dataset/txt/`
3. Extract mel-spectrograms
4. Extract prosody (pitch + energy)
5. Convert text to phonemes
6. Create training batches

---

## Troubleshooting

### Problem: Download Fails

**Solution 1**: Manual download
```bash
# Download links:
# WAV: https://openslr.trmal.net/resources/79/kn_in_male.zip
# TSV: https://openslr.trmal.net/resources/79/line_index_male.tsv

# Place in: data/kannada_tts_dataset/
# Then run: python prepare_dataset.py --no-download
```

**Solution 2**: Check internet connection
```bash
ping openslr.trmal.net
```

### Problem: Extraction Fails

Delete the zip and retry:
```bash
rm data/kannada_tts_dataset/dataset.zip
python prepare_dataset.py --no-download
```

### Problem: No Text Files Created

Check if WAV files exist:
```bash
ls data/kannada_tts_dataset/wav/
# Should show .wav files
```

Check TSV format:
```bash
head -5 data/kannada_tts_dataset/metadata.tsv
```

### Problem: Encoding Issues with Text

Ensure all text files are UTF-8:

**Python:**
```python
# Save with encoding
with open("file.txt", "w", encoding="utf-8") as f:
    f.write("ನಮಸ್ತೆ")
```

**Command line (Linux/Mac):**
```bash
iconv -f ISO-8859-1 -t UTF-8 input.txt > output.txt
```

### Problem: Audio Sample Rate Mismatch

The `organize_dataset.py` script automatically resamples to 22050 Hz:

```bash
python organize_dataset.py \
    --data-dir data/kannada_tts_dataset \
    --audio-source /path/to/audio \
    --sample-rate 22050
```

### Problem: Low File Matching Rate

Check filename patterns:
```bash
# Files should match exactly:
ls data/kannada_tts_dataset/wav/ | head -5
# Output: file_001.wav, file_002.wav, ...

ls data/kannada_tts_dataset/txt/ | head -5
# Output: file_001.txt, file_002.txt, ...
```

---

## Dataset Size Recommendations

| Use Case | Minimum | Recommended | Ideal |
|----------|---------|-------------|-------|
| **Research/Proof of Concept** | 100 samples | 500 samples | 1000 samples |
| **Production Model** | 1000 samples | 5000 samples | 10000+ samples |
| **Multi-speaker** | 500/speaker | 1000/speaker | 2000+/speaker |

---

## Next Steps

1. ✅ Dataset preparation complete
2. 📝 Review config: `configs/hkl_vits_config.json`
3. 🚀 Start training: `python training/train_hkl_vits.py`
4. 📊 Monitor with TensorBoard
5. 🎤 Test inference with checkpoints

---

## Questions?

Refer to:
- [README.md](../README.md) - Project overview
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Architecture details
- [QUICK_START.md](QUICK_START.md) - Quick reference
