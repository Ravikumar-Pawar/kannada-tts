"""
Setup & Validation Script for Kannada TTS FastAPI Application
Checks all dependencies and configurations
"""

import sys
import importlib
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_dependency(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        logger.info(f"✓ {package_name}")
        return True
    except ImportError:
        logger.error(f"✗ {package_name} - NOT INSTALLED")
        return False


def check_all_dependencies():
    """Check all required dependencies"""
    logger.info("\n" + "="*60)
    logger.info("Checking Dependencies")
    logger.info("="*60)
    
    dependencies = [
        ("PyTorch", "torch"),
        ("NumPy", "numpy"),
        ("SciPy", "scipy"),
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
        ("FastAPI", "fastapi"),
        ("Uvicorn", "uvicorn"),
        ("Pydantic", "pydantic"),
    ]
    
    results = []
    for pkg_name, import_name in dependencies:
        results.append(check_dependency(pkg_name, import_name))
    
    return all(results)


def check_project_structure():
    """Check if all required project files exist"""
    import os
    
    logger.info("\n" + "="*60)
    logger.info("Checking Project Structure")
    logger.info("="*60)
    
    required_files = [
        "app.py",
        "run_app.py",
        "requirements.txt",
        "static/index.html",
        "src/__init__.py",
        "src/inference_unified.py",
        "src/model_manager.py",
        "src/metrics_calculator.py",
        "src/hybrid/models/vits_model.py",
        "src/non_hybrid/models/tacotron2_model.py",
    ]
    
    results = []
    for file_path in required_files:
        if os.path.exists(file_path):
            logger.info(f"✓ {file_path}")
            results.append(True)
        else:
            logger.error(f"✗ {file_path} - NOT FOUND")
            results.append(False)
    
    return all(results)


def check_environment():
    """Check Python version and environment"""
    logger.info("\n" + "="*60)
    logger.info("Checking Environment")
    logger.info("="*60)
    
    version_info = sys.version_info
    logger.info(f"Python Version: {version_info.major}.{version_info.minor}.{version_info.micro}")
    
    if version_info.major >= 3 and version_info.minor >= 8:
        logger.info("✓ Python version is compatible")
        return True
    else:
        logger.error("✗ Python 3.8+ is required")
        return False


def main():
    """Run all checks"""
    logger.info("\n")
    logger.info("╔" + "="*58 + "╗")
    logger.info("║ Kannada TTS FastAPI Application - Setup Validator      ║")
    logger.info("╚" + "="*58 + "╝")
    
    checks = [
        ("Environment", check_environment),
        ("Dependencies", check_all_dependencies),
        ("Project Structure", check_project_structure),
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            logger.error(f"Error during {check_name} check: {str(e)}")
            results.append(False)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Summary")
    logger.info("="*60)
    
    if all(results):
        logger.info("✓ All checks passed! Ready to run the application.")
        logger.info("\nTo start the application, run:")
        logger.info("  python run_app.py")
        logger.info("\nOr using uvicorn directly:")
        logger.info("  uvicorn app:app --reload --host 0.0.0.0 --port 8000")
        logger.info("\nThen open: http://localhost:8000")
        return 0
    else:
        logger.error("✗ Some checks failed. Please fix the issues and try again.")
        logger.error("\nCommon solutions:")
        logger.error("1. Install dependencies: pip install -r requirements.txt")
        logger.error("2. Ensure you're in the project root directory")
        logger.error("3. Update Python to 3.8+")
        return 1


if __name__ == "__main__":
    sys.exit(main())
