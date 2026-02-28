#!/usr/bin/env python3
"""
Kannada TTS - Complete Pipeline Demonstration
Run this script to execute the entire TTS workflow in sequence
"""

import subprocess
import sys
import os
from datetime import datetime

def run_command(cmd, description):
    """Run a shell command with error handling"""
    print(f"\n{'='*70}")
    print(f"🚀 {description}")
    print(f"{'='*70}")
    print(f"Command: {cmd}\n")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, cwd=os.getcwd())
        print(f"\n✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error during {description}: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("🎵 KANNADA TTS - COMPLETE PIPELINE")
    print("="*70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Pipeline steps
    steps = [
        {
            "cmd": "python src/data_prep.py",
            "description": "[1/4] Data Preparation - Download & Validate Kannada-M Dataset",
            "skip": False
        },
        {
            "cmd": "python src/train_tacotron.py",
            "description": "[2/4] Model Training - Tacotron2 + HiFiGAN",
            "skip": True  # Skip by default as it takes 24-48 hours
        },
        {
            "cmd": "python src/inference.py",
            "description": "[3/4] Inference - Generate Kannada Speech with Emotion Enhancement",
            "skip": False
        },
        {
            "cmd": "python src/evaluate.py",
            "description": "[4/4] Evaluation - Performance Metrics Analysis",
            "skip": False
        }
    ]
    
    # Print pipeline info
    print("\n📋 Pipeline Configuration:")
    for i, step in enumerate(steps, 1):
        status = "⏭️  SKIP" if step["skip"] else "▶️  RUN"
        print(f"  {status} - {step['description'][:60]}...")
    
    # Ask for confirmation on training
    print("\n" + "-"*70)
    print("⚠️  Note on [2/4] Model Training:")
    print("  - Expected duration: 24-48 hours on GPU")
    print("  - Requires significant GPU memory (recommend T4 or better)")
    print("  - Skipped by default (set skip=False to run)")
    print("-"*70)
    
    # Execute pipeline
    completed = 0
    failed = 0
    
    for step in steps:
        if step["skip"]:
            print(f"\n⏭️  Skipping: {step['description']}")
            print("  To enable: set skip=False in demo.py")
            continue
        
        if run_command(step["cmd"], step["description"]):
            completed += 1
        else:
            failed += 1
            print("\n⚠️  Do you want to continue? (Press Ctrl+C to stop)")
            try:
                input()
            except KeyboardInterrupt:
                print("\n\n⛔ Pipeline interrupted by user")
                break
    
    # Summary
    print("\n" + "="*70)
    print("📊 PIPELINE EXECUTION SUMMARY")
    print("="*70)
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"✅ Successful steps: {completed}")
    print(f"❌ Failed steps: {failed}")
    
    if failed == 0:
        print("\n🎉 All steps completed successfully!")
        print("\n📁 Output files:")
        print("  - data/metadata.csv - Dataset metadata")
        print("  - output/tacotron2/ - Trained Tacotron2")
        print("  - output/inference/ - Generated audio samples")
        print("  - output/evaluation_results.json - Performance metrics")
    else:
        print(f"\n⚠️  {failed} step(s) failed. Check the logs above.")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
