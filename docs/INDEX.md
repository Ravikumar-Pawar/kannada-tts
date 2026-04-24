# 📚 HKL-VITS Documentation Index

Welcome to the HKL-VITS documentation. This folder contains all project documentation organized by topic.

---

## 📍 Start Here

**New to this project?** Start with [../README.md](../README.md) in the root directory.

---

## 📖 Documentation Files

### 1. **QUICK_START.md** ⚡
- **Duration**: 5-10 minutes
- **For**: First-time users, quick setup
- **Contains**:
  - Installation instructions
  - Dataset preparation
  - Training commands
  - Inference examples
  - API usage
  - Troubleshooting

👉 [Read QUICK_START.md](QUICK_START.md)

---

### 2. **PROJECT_SUMMARY.md** 🏗️
- **Duration**: 30 minutes
- **For**: Researchers, technical understanding
- **Contains**:
  - Complete architecture overview
  - Mathematical foundations
  - Component details (8 encoders/modules)
  - Configuration parameters (80+)
  - Usage examples
  - Performance metrics
  - Kannada linguistics background

👉 [Read PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

---

### 3. **IMPLEMENTATION_COMPLETE.md** ✅
- **Duration**: 15 minutes
- **For**: Developers, project status verification
- **Contains**:
  - Complete file inventory
  - Feature implementation checklist
  - Code statistics
  - Quality assurance details
  - Deployment checklist
  - Achievement summary

👉 [Read IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)

---

### 4. **DATASET_PREPARATION.md** 📊
- **Duration**: 20 minutes (reference)
- **For**: Data engineers, dataset management
- **Contains**:
  - Auto-download from OpenSLR
  - Manual dataset organization
  - Dataset structure requirements
  - Audio/text requirements
  - Data validation methods
  - Troubleshooting guide
  - Dataset size recommendations

👉 [Read DATASET_PREPARATION.md](DATASET_PREPARATION.md)

---

### 5. **DATASET_USAGE.md** 🛠️
- **Duration**: 10 minutes
- **For**: Users of dataset tools
- **Contains**:
  - Unified dataset.py tool overview
  - All subcommands (download, prepare, organize, analyze, full)
  - Usage examples and workflows
  - Input requirements
  - Quick reference guide
  - Migration from old scripts

👉 [Read DATASET_USAGE.md](DATASET_USAGE.md)

---

### 6. **DATASET_SCRIPTS_README.md** (Legacy) 📜
- **Status**: For reference only
- **Note**: Functionality consolidated into `dataset.py`
- **Deprecation**: Use DATASET_USAGE.md instead

👉 [Read DATASET_SCRIPTS_README.md](DATASET_SCRIPTS_README.md)

---

## 🎯 Quick Navigation by Use Case

### **I want to get started quickly**
1. Read: [../README.md](../README.md)
2. Read: [QUICK_START.md](QUICK_START.md)
3. Run: `python dataset.py full` (or [read DATASET_USAGE.md](DATASET_USAGE.md) for more options)
4. Run: `python training/train_hkl_vits.py --config configs/hkl_vits_config.json --data_dir data/kannada_tts_dataset`

**Time**: ~1 hour setup + training time

---

### **I need to understand the architecture**
1. Read: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
2. Review: Configuration in `configs/hkl_vits_config.json`
3. Study: Source code in `hkl_vits/` folder

**Time**: ~30 minutes

---

### **I have my own Kannada dataset**
1. Read: [DATASET_USAGE.md](DATASET_USAGE.md)
2. Run: `python dataset.py organize --data-dir data/kannada_tts_dataset --audio-source /my/audio --text-source /my/text --mapping`
3. Run: `python dataset.py analyze --data-dir data/kannada_tts_dataset --save-report`

**Time**: ~15 minutes

---

### **I want to research TTS/linguistics**
1. Read: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
2. Review: Mathematical foundations section
3. Study: `hkl_vits/` source code
4. Review: Loss functions in `hkl_vits/loss_functions.py`

**Time**: ~1-2 hours

---

### **I need to verify implementation status**
1. Read: [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)
2. Check: Feature implementation checklist
3. Review: Code statistics
4. See: Quality assurance details

**Time**: ~10 minutes

---

## 🔗 File Relationships

```
README.md (Entry Point)
    ├── For Overview: Read immediately
    └── Links to:
        ├── QUICK_START.md (5-10 min setup)
        ├── PROJECT_SUMMARY.md (Technical)
        ├── DATASET_USAGE.md (Data tools)
        ├── DATASET_PREPARATION.md (Data guide)
        └── IMPLEMENTATION_COMPLETE.md (Status)

QUICK_START.md
    └── For immediate usage & commands

PROJECT_SUMMARY.md
    └── For understanding architecture & math

DATASET_USAGE.md (NEW - Recommended)
    └── For dataset.py unified tool commands

DATASET_PREPARATION.md + DATASET_SCRIPTS_README.md
    └── For additional dataset context/legacy reference

IMPLEMENTATION_COMPLETE.md
    └── For project status & verification
```

---

## 📊 Documentation Statistics

| Document | Size | Read Time | Audience |
|----------|------|-----------|----------|
| QUICK_START.md | ~400 lines | 5-10 min | Everyone |
| PROJECT_SUMMARY.md | ~600 lines | 20-30 min | Researchers |
| IMPLEMENTATION_COMPLETE.md | ~400 lines | 10-15 min | Developers |
| DATASET_USAGE.md | ~250 lines | 8-10 min | Data users |
| DATASET_PREPARATION.md | ~500 lines | 15-20 min | Data engineers |
| DATASET_SCRIPTS_README.md | ~300 lines | 8-10 min | Reference |
| **Total** | **~2450 lines** | **~70-100 min** | **All** |

---

## 🎓 Learning Path

### Beginner
1. [../README.md](../README.md) - Project overview
2. [QUICK_START.md](QUICK_START.md) - Setup and first run

### Intermediate
1. [QUICK_START.md](QUICK_START.md) - Setup
2. [DATASET_PREPARATION.md](DATASET_PREPARATION.md) - Prepare data
3. [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Understanding

### Advanced
1. [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Architecture
2. [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md) - Details
3. Source code in `hkl_vits/` - Implementation study

### Data Engineers
1. [DATASET_USAGE.md](DATASET_USAGE.md) - dataset.py commands
2. [DATASET_PREPARATION.md](DATASET_PREPARATION.md) - Background/reference
3. Source code in `dataset.py` (unified tool)

---

## 🔍 Find Topics

### Configuration
- See: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Configuration Parameters section
- Edit: `configs/hkl_vits_config.json`

### Model Components
- See: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Component Details section
- Code: `hkl_vits/` folder

### Kannada Phonemes
- See: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Kannada G2P section
- Code: `hkl_vits/kannada_g2p.py`

### Loss Functions
- See: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Loss Functions section
- Code: `hkl_vits/loss_functions.py`

### Dataset Preparation
- See: [DATASET_USAGE.md](DATASET_USAGE.md) (commands)
- See: [DATASET_PREPARATION.md](DATASET_PREPARATION.md) (guide)
- Tool: `dataset.py` (unified)

### Training
- See: [QUICK_START.md](QUICK_START.md) - Train Model section
- Code: `training/train_hkl_vits.py`

### Inference
- See: [QUICK_START.md](QUICK_START.md) - Test Inference section
- Code: `hkl_vits/inference.py`

### Troubleshooting
- See: [QUICK_START.md](QUICK_START.md) - Troubleshooting section
- See: [DATASET_PREPARATION.md](DATASET_PREPARATION.md) - Troubleshooting section

---

## 📝 Documentation Convention

### Symbols Used
- 📋 - Reference guide
- ⚡ - Quick reference
- 🏗️ - Architecture/design
- ✅ - Status/verification
- 📊 - Data-related
- 🛠️ - Tools
- 🎯 - Usage guide
- ❌ - Problems/troubleshooting
- ✨ - Features
- 🔧 - Configuration
- 🚀 - Getting started
- 📚 - Learning

### Code Format
- Command line: `code snippet`
- File paths: `folder/file.py`
- Python: `code blocks`
- Configuration: `json` blocks

---

## 🎯 Common Tasks Quick Links

| Task | Document | Section |
|------|----------|---------|
| Install dependencies | QUICK_START.md | Installation & Setup |
| Download dataset | DATASET_USAGE.md | Quick Commands |
| Organize custom data | DATASET_USAGE.md | Organize Custom Data |
| Analyze dataset | DATASET_USAGE.md | Analyze |
| Complete pipeline | DATASET_USAGE.md | Full Pipeline |
| Train model | QUICK_START.md | Train Model |
| Do inference | QUICK_START.md | Test Inference |
| Understand architecture | PROJECT_SUMMARY.md | Architecture Overview |
| Configure model | PROJECT_SUMMARY.md | Configuration Parameters |
| Troubleshoot issues | QUICK_START.md | Troubleshooting |
| Check implementation | IMPLEMENTATION_COMPLETE.md | Feature Checklist |

---

## ⚡ Super Quick Commands

```bash
# Setup (5 min)
pip install -r requirements.txt

# Prepare dataset (30-60 min)
python dataset.py full

# Or individual steps:
python dataset.py download    # Download from OpenSLR
python dataset.py prepare     # Extract and organize
python dataset.py analyze     # Verify and analyze

# Train (hours/days)
python training/train_hkl_vits.py --config configs/hkl_vits_config.json --data_dir data/kannada_tts_dataset

# Inference (seconds)
python hkl_vits/inference.py --config configs/hkl_vits_config.json --checkpoint checkpoints/model.pt --text "ನಮಸ್ತೆ"
```

---

## 🆘 Need Help?

1. **Setup issues?** → [QUICK_START.md](QUICK_START.md) - Troubleshooting
2. **Dataset problems?** → [DATASET_PREPARATION.md](DATASET_PREPARATION.md) - Troubleshooting
3. **Technical questions?** → [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Architecture section
4. **Not sure where to start?** → [../README.md](../README.md)
5. **Want full reference?** → [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)

---

## 📞 Documentation Updates

**Last Updated**: April 2026  
**Version**: 1.0.0  
**Status**: ✅ Complete and Production Ready

---

**Start with**: [../README.md](../README.md) 🚀

---

*This documentation structure ensures you can find exactly what you need, whether you're a beginner getting started or an expert researcher diving deep.*
