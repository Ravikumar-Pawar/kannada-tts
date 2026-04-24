#!/usr/bin/env python3
# ============================================
# Kannada Dataset Manager for HKL-VITS
# Unified tool for download, prepare, organize, analyze
# ============================================

import os
import sys
import zipfile
import requests
import pandas as pd
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import argparse
import json
from typing import Dict, Optional


# ============================================
# CONFIG
# ============================================

class DatasetConfig:
    """Configuration for dataset operations"""
    
    WAV_ZIP_URL = "https://openslr.trmal.net/resources/79/kn_in_male.zip"
    TSV_URL = "https://openslr.trmal.net/resources/79/line_index_male.tsv"
    
    def __init__(self, base_dir: str = "."):
        self.BASE_DIR = Path(base_dir)
        self.DATA_DIR = self.BASE_DIR / "data"
        self.DATASET_DIR = self.DATA_DIR / "kannada_tts_dataset"
        self.WAV_DIR = self.DATASET_DIR / "wav"
        self.TEXT_DIR = self.DATASET_DIR / "txt"
        self.TEMP_DIR = self.DATASET_DIR / "temp"
        self.ZIP_PATH = self.DATASET_DIR / "dataset.zip"
        self.TSV_PATH = self.DATASET_DIR / "metadata.tsv"
        
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories"""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.DATASET_DIR.mkdir(parents=True, exist_ok=True)
        self.WAV_DIR.mkdir(parents=True, exist_ok=True)
        self.TEXT_DIR.mkdir(parents=True, exist_ok=True)
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# DOWNLOAD
# ============================================

class DatasetDownloader:
    """Downloads audio and metadata from OpenSLR"""
    
    CHUNK_SIZE = 1024 * 1024
    TIMEOUT = 60
    
    def __init__(self, config: DatasetConfig):
        self.config = config
    
    def _download_file(self, url: str, save_path: Path, force: bool = False) -> bool:
        """Download file with progress bar"""
        if save_path.exists() and not force:
            print(f"✓ File already exists: {save_path.name}")
            return True
        
        try:
            print(f"\n⬇️  Downloading: {url}")
            response = requests.get(url, stream=True, timeout=self.TIMEOUT)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            with open(save_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, 
                         desc=save_path.name, leave=True) as pbar:
                    for chunk in response.iter_content(self.CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"✓ Downloaded to: {save_path}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"✗ Download failed: {e}")
            return False
    
    def run(self, force: bool = False) -> bool:
        """Download dataset files"""
        print("\n" + "="*60)
        print("DOWNLOADING DATASET")
        print("="*60)
        
        zip_success = self._download_file(self.config.WAV_ZIP_URL, self.config.ZIP_PATH, force)
        tsv_success = self._download_file(self.config.TSV_URL, self.config.TSV_PATH, force)
        
        if not (zip_success and tsv_success):
            print("\n✗ Some downloads failed!")
            return False
        
        print("\n✓ All downloads completed!")
        return True


# ============================================
# EXTRACT & PREPARE
# ============================================

class DatasetPreparer:
    """Extracts and prepares dataset"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
    
    def extract(self, force: bool = False) -> bool:
        """Extract WAV files from ZIP"""
        print("\n" + "="*60)
        print("EXTRACTING AUDIO FILES")
        print("="*60)
        
        if not self.config.ZIP_PATH.exists():
            print(f"✗ ZIP file not found: {self.config.ZIP_PATH}")
            return False
        
        wav_count = len(list(self.config.WAV_DIR.glob('*.wav')))
        if wav_count > 0 and not force:
            print(f"✓ WAV files already extracted ({wav_count} files)")
            return True
        
        try:
            print(f"\n⬇️  Extracting from: {self.config.ZIP_PATH.name}")
            
            with zipfile.ZipFile(self.config.ZIP_PATH, 'r') as zip_ref:
                all_files = zip_ref.namelist()
                wav_files = [f for f in all_files if f.endswith('.wav')]
                print(f"Found {len(wav_files)} WAV files in archive")
                
                for file_info in tqdm(zip_ref.infolist(), desc="Extracting", leave=True):
                    if file_info.filename.endswith('.wav'):
                        zip_ref.extract(file_info, self.config.TEMP_DIR)
            
            print("\n📦 Organizing extracted files...")
            moved_count = 0
            for wav_file in self.config.TEMP_DIR.rglob('*.wav'):
                dest = self.config.WAV_DIR / wav_file.name
                if not dest.exists():
                    shutil.move(str(wav_file), str(dest))
                    moved_count += 1
            
            shutil.rmtree(self.config.TEMP_DIR)
            self.config.TEMP_DIR.mkdir(exist_ok=True)
            
            print(f"✓ Extracted and organized {moved_count} WAV files")
            print(f"✓ WAV files location: {self.config.WAV_DIR}")
            return True
        except Exception as e:
            print(f"✗ Extraction failed: {e}")
            return False
    
    def prepare_text(self, force: bool = False) -> int:
        """Create text files from TSV metadata"""
        print("\n" + "="*60)
        print("PREPARING TEXT FILES")
        print("="*60)
        
        if not self.config.TSV_PATH.exists():
            print(f"✗ TSV file not found: {self.config.TSV_PATH}")
            return 0
        
        try:
            print(f"\n📖 Reading: {self.config.TSV_PATH.name}")
            df = pd.read_csv(self.config.TSV_PATH, sep='\t', 
                           header=None, names=['wav_file', 'text'])
            print(f"Found {len(df)} text entries in metadata")
            
            created_count = 0
            skipped_count = 0
            
            print("\n✍️  Creating text files...")
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
                wav_file = str(row['wav_file']).strip()
                text = str(row['text']).strip()
                base_name = Path(wav_file).stem
                text_file_path = self.config.TEXT_DIR / f"{base_name}.txt"
                wav_path = self.config.WAV_DIR / f"{base_name}.wav"
                
                if not wav_path.exists():
                    skipped_count += 1
                    continue
                
                if text_file_path.exists() and not force:
                    skipped_count += 1
                    continue
                
                with open(text_file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                created_count += 1
            
            print(f"\n✓ Created {created_count} text files")
            if skipped_count > 0:
                print(f"⏭️  Skipped {skipped_count} existing/missing files")
            
            return created_count
        except Exception as e:
            print(f"✗ Text preparation failed: {e}")
            return 0


# ============================================
# ORGANIZE
# ============================================

class DatasetOrganizer:
    """Organizes custom Kannada datasets"""
    
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.wav_dir = self.dataset_dir / "wav"
        self.txt_dir = self.dataset_dir / "txt"
        self.wav_dir.mkdir(parents=True, exist_ok=True)
        self.txt_dir.mkdir(parents=True, exist_ok=True)
    
    def organize_audio(self, source_dir: str, sample_rate: int = 22050) -> int:
        """Organize audio files with resampling"""
        source_path = Path(source_dir)
        
        if not source_path.exists():
            print(f"✗ Source directory not found: {source_dir}")
            return 0
        
        print(f"\n🔍 Scanning: {source_dir}")
        wav_files = list(source_path.rglob('*.wav')) + list(source_path.rglob('*.mp3'))
        
        if not wav_files:
            print("✗ No audio files found!")
            return 0
        
        print(f"Found {len(wav_files)} audio files\n")
        
        processed_count = 0
        for audio_file in tqdm(wav_files, desc="Processing"):
            try:
                y, sr = librosa.load(str(audio_file), sr=None)
                if sr != sample_rate:
                    y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
                
                out_path = self.wav_dir / audio_file.stem
                out_path = out_path.with_suffix('.wav')
                sf.write(str(out_path), y, sample_rate)
                processed_count += 1
            except Exception as e:
                print(f"⚠️  Failed to process {audio_file.name}: {e}")
        
        print(f"\n✓ Processed {processed_count} audio files")
        return processed_count
    
    def organize_text(self, source_dir: str) -> int:
        """Organize text files"""
        source_path = Path(source_dir)
        
        if not source_path.exists():
            print(f"✗ Source directory not found: {source_dir}")
            return 0
        
        print(f"\n📖 Scanning: {source_dir}")
        txt_files = list(source_path.glob('*.txt'))
        
        if not txt_files:
            print("✗ No text files found!")
            return 0
        
        print(f"Found {len(txt_files)} text files\n")
        wav_stems = {f.stem for f in self.wav_dir.glob('*.wav')}
        
        processed_count = 0
        for txt_file in tqdm(txt_files, desc="Processing"):
            if txt_file.stem not in wav_stems:
                print(f"⚠️  No matching WAV for: {txt_file.name}")
                continue
            
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                out_path = self.txt_dir / txt_file.name
                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                processed_count += 1
            except Exception as e:
                print(f"⚠️  Failed to process {txt_file.name}: {e}")
        
        print(f"\n✓ Processed {processed_count} text files")
        return processed_count
    
    def create_mapping(self, output_file: str = "dataset_mapping.txt"):
        """Create dataset mapping file"""
        mapping_path = self.dataset_dir / output_file
        wav_files = sorted(self.wav_dir.glob('*.wav'))
        
        with open(mapping_path, 'w', encoding='utf-8') as f:
            f.write("Sample ID\t\tWAV File\t\tText File\n")
            f.write("="*80 + "\n")
            
            for wav_file in wav_files:
                txt_file = self.txt_dir / (wav_file.stem + '.txt')
                status = "✓" if txt_file.exists() else "✗"
                f.write(f"{wav_file.stem}\t{wav_file.name}\t{status}\n")
        
        print(f"\n✓ Dataset mapping saved to: {mapping_path}")


# ============================================
# ANALYZE
# ============================================

class DatasetAnalyzer:
    """Analyzes and validates datasets"""
    
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.wav_dir = self.dataset_dir / "wav"
        self.txt_dir = self.dataset_dir / "txt"
        self.stats = {'issues': []}
    
    def run(self, save_report: bool = False) -> bool:
        """Run complete analysis"""
        print("\n" + "█"*60)
        print("█  KANNADA TTS DATASET ANALYZER")
        print("█"*60)
        
        if not self._check_structure():
            return False
        
        wav_files, txt_files, matched_stems = self._analyze_matching()
        
        if matched_stems:
            self._analyze_audio(wav_files, matched_stems)
            self._analyze_text(txt_files, matched_stems)
            self._show_samples(wav_files, txt_files, matched_stems)
        
        report = self._generate_report()
        
        if save_report:
            report_path = self.dataset_dir / "analysis_report.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n✓ Report saved to: {report_path}")
        
        return len(self.stats['issues']) == 0
    
    def _check_structure(self) -> bool:
        """Check dataset structure"""
        print("\n" + "="*60)
        print("DATASET STRUCTURE CHECK")
        print("="*60)
        
        if not self.wav_dir.exists() or not self.txt_dir.exists():
            print(f"✗ Missing directories")
            return False
        
        print(f"✓ WAV directory: {self.wav_dir}")
        print(f"✓ TXT directory: {self.txt_dir}")
        return True
    
    def _analyze_matching(self):
        """Analyze file matching"""
        print("\n" + "="*60)
        print("FILE MATCHING ANALYSIS")
        print("="*60)
        
        wav_files = {f.stem: f for f in self.wav_dir.glob('*.wav')}
        txt_files = {f.stem: f for f in self.txt_dir.glob('*.txt')}
        
        wav_stems = set(wav_files.keys())
        txt_stems = set(txt_files.keys())
        matched = wav_stems & txt_stems
        
        print(f"\n📊 Counts:")
        print(f"   WAV files: {len(wav_files)}")
        print(f"   TXT files: {len(txt_files)}")
        print(f"   Matched pairs: {len(matched)}")
        print(f"   Matching rate: {100*len(matched)/max(len(wav_stems), 1):.1f}%")
        
        return wav_files, txt_files, matched
    
    def _analyze_audio(self, wav_files: Dict, matched_stems: set):
        """Analyze audio properties"""
        print("\n" + "="*60)
        print("AUDIO ANALYSIS")
        print("="*60)
        
        sample_rates, durations, file_sizes = {}, [], []
        
        print(f"\n🔍 Analyzing {len(matched_stems)} audio files...")
        for stem in tqdm(matched_stems, desc="Audio"):
            try:
                wav_path = wav_files[stem]
                file_size_mb = wav_path.stat().st_size / (1024 * 1024)
                file_sizes.append(file_size_mb)
                
                info = sf.info(str(wav_path))
                sr = info.samplerate
                sample_rates[sr] = sample_rates.get(sr, 0) + 1
                durations.append(info.duration)
            except Exception as e:
                self.stats['issues'].append(f"Error reading {wav_path.name}")
        
        if durations:
            print(f"\n⏱️  Duration Statistics:")
            print(f"   Min: {min(durations):.2f}s, Max: {max(durations):.2f}s")
            print(f"   Mean: {np.mean(durations):.2f}s, Median: {np.median(durations):.2f}s")
        
        if file_sizes:
            print(f"\n💾 File Size Statistics:")
            print(f"   Total size: {sum(file_sizes):.2f} MB")
            print(f"   Mean size: {np.mean(file_sizes):.2f} MB")
        
        if sample_rates:
            print(f"\n🎵 Sample Rates:")
            for sr, count in sorted(sample_rates.items(), key=lambda x: -x[1]):
                pct = 100 * count / len(matched_stems)
                status = "✓" if sr == 22050 else "⚠️"
                print(f"   {sr} Hz: {count} files ({pct:.1f}%) {status}")
                if sr != 22050:
                    self.stats['issues'].append(f"{count} files at {sr}Hz")
    
    def _analyze_text(self, txt_files: Dict, matched_stems: set):
        """Analyze text properties"""
        print("\n" + "="*60)
        print("TEXT ANALYSIS")
        print("="*60)
        
        word_counts, encoding_issues = [], 0
        kannada_count = 0
        
        print(f"\n🔍 Analyzing {len(matched_stems)} text files...")
        for stem in tqdm(matched_stems, desc="Text"):
            try:
                with open(txt_files[stem], 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                word_counts.append(len(text.split()))
                kannada_count += sum(1 for c in text if 0x0C80 <= ord(c) <= 0x0CFF)
            except UnicodeDecodeError:
                encoding_issues += 1
                self.stats['issues'].append(f"Encoding error")
            except Exception as e:
                self.stats['issues'].append(f"Text read error")
        
        if word_counts:
            print(f"\n📝 Word Count Statistics:")
            print(f"   Min: {min(word_counts)}, Max: {max(word_counts)}")
            print(f"   Mean: {np.mean(word_counts):.1f}, Median: {np.median(word_counts):.1f}")
        
        if encoding_issues > 0:
            print(f"\n✗ Encoding Issues: {encoding_issues} files")
        
        print(f"\n🇮🇳 Kannada Characters: {kannada_count} total")
    
    def _show_samples(self, wav_files: Dict, txt_files: Dict, matched_stems: set, count: int = 3):
        """Show sample data"""
        print("\n" + "="*60)
        print(f"SAMPLE DATA ({min(count, len(matched_stems))} samples)")
        print("="*60)
        
        for i, stem in enumerate(sorted(matched_stems)[:count], 1):
            print(f"\n📋 Sample {i}: {stem}")
            try:
                info = sf.info(str(wav_files[stem]))
                print(f"   🎵 Audio: {info.samplerate}Hz, {info.duration:.2f}s")
            except:
                pass
            
            try:
                with open(txt_files[stem], 'r', encoding='utf-8') as f:
                    text = f.read().strip()[:50]
                    print(f"   📝 Text: {text}...")
            except:
                pass
    
    def _generate_report(self) -> str:
        """Generate analysis report"""
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        
        report = f"""
DATASET ANALYSIS REPORT
{'='*60}

STATUS: {"✓ READY" if len(self.stats['issues']) == 0 else "⚠️ ISSUES FOUND"}

ISSUES: {len(self.stats['issues'])}
"""
        if self.stats['issues']:
            for i, issue in enumerate(self.stats['issues'][:5], 1):
                report += f"  {i}. {issue}\n"
        else:
            report += "  ✓ No issues detected!\n"
        
        return report


# ============================================
# CLI
# ============================================

def main():
    """Main CLI"""
    parser = argparse.ArgumentParser(
        description='Kannada Dataset Manager for HKL-VITS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
SUBCOMMANDS:
  download    Download dataset from OpenSLR
  prepare     Extract and prepare downloaded dataset
  organize    Organize custom Kannada dataset
  analyze     Analyze and validate dataset
  full        Download + prepare (complete OpenSLR pipeline)

EXAMPLES:
  # Download from OpenSLR
  python dataset.py download
  
  # Extract and prepare
  python dataset.py prepare
  
  # Complete pipeline
  python dataset.py full
  
  # Organize custom data
  python dataset.py organize \\
    --data-dir data/kannada_tts_dataset \\
    --audio-source /path/to/audio \\
    --text-source /path/to/text
  
  # Analyze dataset
  python dataset.py analyze --data-dir data/kannada_tts_dataset --save-report
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Download command
    dl = subparsers.add_parser('download', help='Download dataset from OpenSLR')
    dl.add_argument('--force', action='store_true', help='Force re-download')
    
    # Prepare command
    pr = subparsers.add_parser('prepare', help='Extract and prepare dataset')
    pr.add_argument('--force', action='store_true', help='Force re-processing')
    
    # Organize command
    og = subparsers.add_parser('organize', help='Organize custom dataset')
    og.add_argument('--data-dir', required=True, help='Dataset directory')
    og.add_argument('--audio-source', help='Source audio directory')
    og.add_argument('--text-source', help='Source text directory')
    og.add_argument('--sample-rate', type=int, default=22050, help='Target sample rate')
    og.add_argument('--mapping', action='store_true', help='Generate mapping file')
    
    # Analyze command
    an = subparsers.add_parser('analyze', help='Analyze dataset')
    an.add_argument('--data-dir', required=True, help='Dataset directory')
    an.add_argument('--save-report', action='store_true', help='Save report to file')
    
    # Full command
    fl = subparsers.add_parser('full', help='Full OpenSLR pipeline')
    fl.add_argument('--force', action='store_true', help='Force re-processing')
    fl.add_argument('--skip-analyze', action='store_true', help='Skip analysis')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Download
    if args.command == 'download':
        config = DatasetConfig()
        downloader = DatasetDownloader(config)
        success = downloader.run(args.force)
        return 0 if success else 1
    
    # Prepare
    elif args.command == 'prepare':
        config = DatasetConfig()
        preparer = DatasetPreparer(config)
        extract_ok = preparer.extract(args.force)
        if extract_ok:
            text_count = preparer.prepare_text(args.force)
            success = text_count > 0
        else:
            success = False
        return 0 if success else 1
    
    # Organize
    elif args.command == 'organize':
        organizer = DatasetOrganizer(args.data_dir)
        if args.audio_source:
            organizer.organize_audio(args.audio_source, args.sample_rate)
        if args.text_source:
            organizer.organize_text(args.text_source)
        if args.mapping or (args.audio_source or args.text_source):
            organizer.create_mapping()
        print("\n✅ Dataset organization complete!")
        return 0
    
    # Analyze
    elif args.command == 'analyze':
        analyzer = DatasetAnalyzer(args.data_dir)
        success = analyzer.run(args.save_report)
        return 0 if success else 1
    
    # Full
    elif args.command == 'full':
        print("\n" + "█"*60)
        print("█  FULL DATASET PIPELINE")
        print("█"*60)
        
        config = DatasetConfig()
        
        # Download
        print("\n[1/3] DOWNLOADING...")
        downloader = DatasetDownloader(config)
        if not downloader.run(args.force):
            return 1
        
        # Prepare
        print("\n[2/3] PREPARING...")
        preparer = DatasetPreparer(config)
        if not preparer.extract(args.force):
            return 1
        text_count = preparer.prepare_text(args.force)
        if text_count == 0:
            return 1
        
        # Analyze
        if not args.skip_analyze:
            print("\n[3/3] ANALYZING...")
            analyzer = DatasetAnalyzer(config.DATASET_DIR)
            analyzer.run(False)
        
        print("\n" + "█"*60)
        print("█  ✅ PIPELINE COMPLETE!")
        print("█"*60)
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
