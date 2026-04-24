# Dataset Preparation Scripts - Quick Reference

This document provides a quick overview of the dataset preparation tools created for HKL-VITS.

## 📁 Dataset Preparation Tools

### 1. **prepare_dataset.py** - Automatic Dataset Pipeline
**Location**: Root directory  
**Purpose**: Download and prepare Kannada TTS dataset from OpenSLR

**Features**:
- ✅ Automatic download from OpenSLR
- ✅ ZIP extraction with progress tracking
- ✅ Metadata processing
- ✅ Text file generation
- ✅ Full validation

**Usage**:
```bash
# Full pipeline
python prepare_dataset.py

# Skip download
python prepare_dataset.py --no-download

# Force re-process
python prepare_dataset.py --force

# View help
python prepare_dataset.py --help
```

**Output**:
- `data/kannada_tts_dataset/wav/` - Audio files (22050 Hz WAV)
- `data/kannada_tts_dataset/txt/` - Text transcriptions (UTF-8)
- `data/kannada_tts_dataset/metadata.tsv` - Original metadata

**Time**: ~30-60 minutes (depends on internet)

---

### 2. **organize_dataset.py** - Manual Dataset Organization
**Location**: Root directory  
**Purpose**: Organize your own Kannada dataset

**Features**:
- ✅ Batch process local audio files
- ✅ Automatic resampling to 22050 Hz
- ✅ Filename matching for audio-text pairs
- ✅ Generate mapping file
- ✅ UTF-8 encoding verification

**Usage**:
```bash
# Organize audio files only
python organize_dataset.py \
    --data-dir data/kannada_tts_dataset \
    --audio-source /path/to/my_audio

# Organize text files only
python organize_dataset.py \
    --data-dir data/kannada_tts_dataset \
    --text-source /path/to/my_text

# Organize both + generate mapping
python organize_dataset.py \
    --data-dir data/kannada_tts_dataset \
    --audio-source /path/to/audio \
    --text-source /path/to/text \
    --mapping
```

**Input Requirements**:
- Audio: .wav or .mp3 files (any sample rate)
- Text: .txt files (any encoding)
- Filenames: Must match (e.g., sample_001.wav ↔ sample_001.txt)

**Output**:
- Standardized dataset in `data/kannada_tts_dataset/`
- Optional: `dataset_mapping.txt` showing file pairs

---

### 3. **analyze_dataset.py** - Dataset Analysis & Validation
**Location**: Root directory  
**Purpose**: Inspect and validate your dataset

**Features**:
- ✅ File matching analysis
- ✅ Audio properties (duration, sample rate, size)
- ✅ Text analysis (word count, Kannada characters)
- ✅ Sample data preview
- ✅ Issue detection and reporting
- ✅ Detailed statistics

**Usage**:
```bash
# Quick analysis
python analyze_dataset.py --data-dir data/kannada_tts_dataset

# Save report
python analyze_dataset.py \
    --data-dir data/kannada_tts_dataset \
    --save-report
```

**Output**:
- Console report with statistics
- Optional: `analysis_report.txt` file

---

## 🚀 Workflow Examples

### Example 1: Download OpenSLR Dataset

```bash
# Step 1: Download and prepare
python prepare_dataset.py

# Step 2: Analyze dataset
python analyze_dataset.py --data-dir data/kannada_tts_dataset

# Step 3: Start training
python training/train_hkl_vits.py \
    --config configs/hkl_vits_config.json \
    --data_dir data/kannada_tts_dataset
```

---

### Example 2: Use Your Own Dataset

```bash
# Step 1: Organize your data
python organize_dataset.py \
    --data-dir data/kannada_tts_dataset \
    --audio-source /path/to/my_audio \
    --text-source /path/to/my_text \
    --mapping

# Step 2: Validate organization
python analyze_dataset.py --data-dir data/kannada_tts_dataset --save-report

# Step 3: Train
python training/train_hkl_vits.py \
    --config configs/hkl_vits_config.json \
    --data_dir data/kannada_tts_dataset
```

---

### Example 3: Combine Multiple Datasets

```bash
# Create main dataset
python prepare_dataset.py

# Add more data
python organize_dataset.py \
    --data-dir data/kannada_tts_dataset \
    --audio-source /path/to/additional_audio \
    --text-source /path/to/additional_text

# Verify combined dataset
python analyze_dataset.py --data-dir data/kannada_tts_dataset --save-report
```

---

## 📋 Dataset Structure Reference

After preparation:

```
data/
└── kannada_tts_dataset/
    ├── wav/                   # Audio files (22050 Hz)
    │   ├── sample_001.wav
    │   ├── sample_002.wav
    │   └── ... (1000+ files)
    ├── txt/                   # Text transcriptions (UTF-8)
    │   ├── sample_001.txt     # "ನಮಸ್ತೆ"
    │   ├── sample_002.txt     # "ಧನ್ಯವಾದ"
    │   └── ... (must match)
    ├── metadata.tsv           # OpenSLR metadata
    └── analysis_report.txt    # Generated analysis
```

**Critical Requirements**:
- ✅ Filenames must match exactly (except extension)
- ✅ All audio must be 22050 Hz
- ✅ All text must be UTF-8 encoded
- ✅ Text must be in Kannada script

---

## ⚙️ Installation & Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python analyze_dataset.py --data-dir data
```

---

## 🔧 Common Commands

```bash
# Download & Setup
python prepare_dataset.py && python analyze_dataset.py --data-dir data/kannada_tts_dataset

# Add Custom Data
python organize_dataset.py \
    --data-dir data/kannada_tts_dataset \
    --audio-source /my/audio \
    --text-source /my/text

# Generate Analysis Report
python analyze_dataset.py --data-dir data/kannada_tts_dataset --save-report

# Check Dataset Matching
python analyze_dataset.py --data-dir data/kannada_tts_dataset
```

---

## ⚠️ Troubleshooting

### Download Issues
```bash
# Check internet
ping openslr.trmal.net

# Manual download retry
rm data/kannada_tts_dataset/dataset.zip
python prepare_dataset.py --no-download
```

### Audio Format Issues
```bash
python organize_dataset.py \
    --data-dir data/kannada_tts_dataset \
    --audio-source /path/to/audio \
    --sample-rate 22050
```

### Text Encoding Issues
```bash
python analyze_dataset.py --data-dir data/kannada_tts_dataset
# Look for "Encoding Issues"
```

### File Matching Problems
```bash
python analyze_dataset.py --data-dir data/kannada_tts_dataset
# Check "Matching rate" in output
```

---

## 📊 Dataset Recommendations

| Scenario | Size | Time | Approach |
|----------|------|------|----------|
| **Quick Test** | 50-100 | <10 min | Manual sample |
| **Research** | 500-1000 | 1-2 hours | OpenSLR download |
| **Production** | 5000+ | Hours/days | OpenSLR + custom |
| **Multi-speaker** | 1000+/speaker | Days | Multiple datasets |

---

## 📚 Related Documentation

- [README.md](../README.md) - Project overview
- [QUICK_START.md](QUICK_START.md) - Quick reference
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Architecture
- [DATASET_PREPARATION.md](DATASET_PREPARATION.md) - Detailed guide

---

## 🎯 Next Steps

1. ✅ **Install**: `pip install -r requirements.txt`
2. ✅ **Prepare**: `python prepare_dataset.py`
3. ✅ **Analyze**: `python analyze_dataset.py --data-dir data/kannada_tts_dataset`
4. 🚀 **Train**: `python training/train_hkl_vits.py --config configs/hkl_vits_config.json --data_dir data/kannada_tts_dataset`

---

**Version**: 1.0.0 | **Date**: April 2026
