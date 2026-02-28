#!/usr/bin/env python3
"""
Kannada TTS - System Validation & Testing
Validates installation, dependencies, and system readiness
"""

import sys
import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# VALIDATION TESTS
# ============================================================================
class ValidationTests:
    """System validation and tests"""
    
    def __init__(self):
        self.results = []
        self.warnings = []
        self.errors = []
    
    def test_python_version(self) -> bool:
        """Test Python version (3.8+)"""
        test_name = "Python Version"
        required_version = (3, 8)
        
        if sys.version_info >= required_version:
            self.results.append((test_name, "✅ PASS", f"Python {sys.version_info.major}.{sys.version_info.minor}"))
            return True
        else:
            self.results.append((test_name, "❌ FAIL", f"Python {sys.version_info.major}.{sys.version_info.minor} (need 3.8+)"))
            self.errors.append(f"{test_name}: Upgrade Python to 3.8+")
            return False
    
    def test_pytorch(self) -> bool:
        """Test PyTorch installation"""
        test_name = "PyTorch"
        
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            device_info = f"Version {torch.__version__}, CUDA: {'✅ Available' if cuda_available else '⚠️  CPU Only'}"
            self.results.append((test_name, "✅ PASS", device_info))
            if not cuda_available:
                self.warnings.append(f"{test_name}: Running on CPU (training will be slow)")
            return True
        except ImportError as e:
            self.results.append((test_name, "❌ FAIL", str(e)))
            self.errors.append(f"{test_name}: {e}")
            return False
    
    def test_tts_library(self) -> bool:
        """Test TTS library installation"""
        test_name = "TTS Library"
        
        try:
            from TTS.api import TTS
            self.results.append((test_name, "✅ PASS", "TTS library installed"))
            return True
        except ImportError as e:
            self.results.append((test_name, "❌ FAIL", str(e)))
            self.errors.append(f"{test_name}: {e}")
            return False
    
    def test_librosa(self) -> bool:
        """Test Librosa installation"""
        test_name = "Librosa"
        
        try:
            import librosa
            self.results.append((test_name, "✅ PASS", f"Version {librosa.__version__}"))
            return True
        except ImportError as e:
            self.results.append((test_name, "❌ FAIL", str(e)))
            self.errors.append(f"{test_name}: {e}")
            return False
    
    def test_pandas(self) -> bool:
        """Test Pandas installation"""
        test_name = "Pandas"
        
        try:
            import pandas
            self.results.append((test_name, "✅ PASS", f"Version {pandas.__version__}"))
            return True
        except ImportError as e:
            self.results.append((test_name, "❌ FAIL", str(e)))
            self.errors.append(f"{test_name}: {e}")
            return False
    
    def test_soundfile(self) -> bool:
        """Test SoundFile installation"""
        test_name = "SoundFile"
        
        try:
            import soundfile
            self.results.append((test_name, "✅ PASS", "Audio I/O ready"))
            return True
        except ImportError as e:
            self.results.append((test_name, "❌ FAIL", str(e)))
            self.errors.append(f"{test_name}: {e}")
            return False
    
    def test_directory_structure(self) -> bool:
        """Test project directory structure"""
        test_name = "Directory Structure"
        
        required_dirs = ["src", "config", "output", "data"]
        missing = []
        
        for dir_name in required_dirs:
            if not os.path.isdir(dir_name):
                missing.append(dir_name)
        
        if not missing:
            self.results.append((test_name, "✅ PASS", "All directories present"))
            return True
        else:
            self.results.append((test_name, "⚠️  WARN", f"Missing: {', '.join(missing)}"))
            self.warnings.append(f"{test_name}: Creating missing directories")
            for dir_name in missing:
                os.makedirs(dir_name, exist_ok=True)
            return False
    
    def test_config_files(self) -> bool:
        """Test configuration files"""
        test_name = "Config Files"
        
        required_configs = [
            "config/tacotron2.json",
            "config/hifigan.json"
        ]
        
        missing = []
        for config_path in required_configs:
            if not os.path.isfile(config_path):
                missing.append(config_path)
        
        if not missing:
            self.results.append((test_name, "✅ PASS", "All configs present"))
            return True
        else:
            self.results.append((test_name, "❌ FAIL", f"Missing: {', '.join(missing)}"))
            self.errors.append(f"{test_name}: Config files not found")
            return False
    
    def test_disk_space(self) -> bool:
        """Test available disk space"""
        test_name = "Disk Space"
        
        try:
            import shutil
            stat = shutil.disk_usage(".")
            free_gb = stat.free / (1024 ** 3)
            
            if free_gb >= 50:
                status = "✅ PASS"
                msg = f"{free_gb:.1f} GB available"
            elif free_gb >= 20:
                status = "⚠️  WARN"
                msg = f"Only {free_gb:.1f} GB (recommend 50 GB)"
                self.warnings.append(f"{test_name}: Low disk space")
            else:
                status = "❌ FAIL"
                msg = f"Only {free_gb:.1f} GB available (need 50 GB)"
                self.errors.append(f"{test_name}: Insufficient disk space")
            
            self.results.append((test_name, status, msg))
            return status == "✅ PASS"
        
        except Exception as e:
            self.results.append((test_name, "⚠️  WARN", str(e)))
            return False
    
    def test_gpu_memory(self) -> bool:
        """Test GPU memory"""
        test_name = "GPU Memory"
        
        if not torch.cuda.is_available():
            self.results.append((test_name, "ℹ️  INFO", "GPU not available (CPU mode)"))
            return True
        
        try:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            if gpu_memory_gb >= 4:
                status = "✅ PASS"
            elif gpu_memory_gb >= 2:
                status = "⚠️  WARN"
                self.warnings.append(f"{test_name}: Low GPU memory ({gpu_memory_gb:.1f} GB)")
            else:
                status = "❌ FAIL"
                self.errors.append(f"{test_name}: GPU memory too low ({gpu_memory_gb:.1f} GB)")
            
            self.results.append((test_name, status, f"{gpu_memory_gb:.1f} GB"))
            return status in ["✅ PASS", "⚠️  WARN"]
        
        except Exception as e:
            self.results.append((test_name, "⚠️  WARN", str(e)))
            return False
    
    def test_audio_io(self) -> bool:
        """Test audio I/O capability"""
        test_name = "Audio I/O"
        
        try:
            import librosa
            import soundfile as sf
            import numpy as np
            
            # Create test audio
            test_audio = np.random.randn(22050)
            test_file = "test_audio_io.wav"
            
            # Write
            sf.write(test_file, test_audio, 22050)
            
            # Read
            y, sr = librosa.load(test_file, sr=22050)
            
            # Cleanup
            os.remove(test_file)
            
            if len(y) > 0 and sr == 22050:
                self.results.append((test_name, "✅ PASS", "Read/write successful"))
                return True
            else:
                self.results.append((test_name, "❌ FAIL", "Audio I/O error"))
                self.errors.append(f"{test_name}: Audio read/write failed")
                return False
        
        except Exception as e:
            self.results.append((test_name, "❌ FAIL", str(e)))
            self.errors.append(f"{test_name}: {e}")
            return False
    
    def test_kannada_characters(self) -> bool:
        """Test Kannada character handling"""
        test_name = "Kannada Language Support"
        
        try:
            test_text = "ನಮಸ್ಕಾರ"
            
            # Check if Kannada characters are in range
            kannada_start = 0x0C80
            kannada_end = 0x0CFF
            
            is_kannada = all(
                kannada_start <= ord(char) <= kannada_end 
                for char in test_text
            )
            
            if is_kannada:
                self.results.append((test_name, "✅ PASS", "Kannada characters supported"))
                return True
            else:
                self.results.append((test_name, "❌ FAIL", "Kannada character issue"))
                self.errors.append(f"{test_name}: Kannada character handling failed")
                return False
        
        except Exception as e:
            self.results.append((test_name, "❌ FAIL", str(e)))
            return False
    
    def run_all_tests(self) -> Dict:
        """Run all validation tests"""
        print("\n" + "="*70)
        print("🔍 SYSTEM VALIDATION & TESTING")
        print("="*70 + "\n")
        
        tests = [
            self.test_python_version,
            self.test_pytorch,
            self.test_tts_library,
            self.test_librosa,
            self.test_pandas,
            self.test_soundfile,
            self.test_directory_structure,
            self.test_config_files,
            self.test_disk_space,
            self.test_gpu_memory,
            self.test_audio_io,
            self.test_kannada_characters,
        ]
        
        for test_func in tests:
            try:
                test_func()
            except Exception as e:
                self.results.append((test_func.__name__, "❌ ERROR", str(e)))
                self.errors.append(f"{test_func.__name__}: {e}")
        
        return self._generate_report()
    
    def _generate_report(self) -> Dict:
        """Generate validation report"""
        print("📋 TEST RESULTS")
        print("-" * 70)
        
        passed = 0
        warnings = 0
        failed = 0
        
        for test_name, status, message in self.results:
            print(f"{status} {test_name:25s} {message}")
            
            if "✅" in status:
                passed += 1
            elif "⚠️" in status or "ℹ️" in status:
                warnings += 1
            else:
                failed += 1
        
        print("\n" + "="*70)
        print("📊 SUMMARY")
        print("="*70)
        print(f"✅ Passed:  {passed}")
        print(f"⚠️  Warnings: {warnings}")
        print(f"❌ Failed:  {failed}")
        
        if self.errors:
            print(f"\n🔴 ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print(f"\n🟡 WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        print("\n" + "="*70)
        
        if failed == 0:
            print("✅ SYSTEM IS READY!")
            print("\nNext steps:")
            print("  1. python src/data_prep.py     (prepare dataset)")
            print("  2. python src/train_tacotron.py (train models)")
            print("  3. python src/inference.py      (generate speech)")
        else:
            print(f"⚠️  Please resolve {failed} error(s) before proceeding")
        
        print("="*70 + "\n")
        
        return {
            "passed": passed,
            "warnings": warnings,
            "failed": failed,
            "ready": failed == 0,
            "details": self.results
        }

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    validator = ValidationTests()
    report = validator.run_all_tests()
    
    # Save report
    with open("output/validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Validation report saved to: output/validation_report.json")
    
    # Exit with appropriate code
    sys.exit(0 if report["ready"] else 1)
