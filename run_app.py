#!/usr/bin/env python
"""
Kannada TTS Web Application Launcher
Starts the FastAPI server with Kannada Text-to-Speech capabilities
"""

import os
import sys
import logging
try:
    import uvicorn
except ImportError:
    uvicorn = None
    # will inform user later

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Start the FastAPI application"""
    
    logger.info("=" * 60)
    logger.info("Kannada Text-to-Speech System - Web Application")
    logger.info("=" * 60)
    
    # Check if app.py exists
    if not os.path.exists("app.py"):
        logger.error("app.py not found in current directory")
        sys.exit(1)
    
    # Start the server
    logger.info("Starting FastAPI server...")
    logger.info("Open your browser and go to: http://localhost:8000")
    logger.info("\nEndpoints:")
    logger.info("  UI:        http://localhost:8000/")
    logger.info("  API Docs:  http://localhost:8000/docs")
    logger.info("  Health:    http://localhost:8000/health")
    logger.info("\nPress Ctrl+C to stop the server")
    logger.info("=" * 60)
    
    if uvicorn is None:
        logger.error("uvicorn is not installed. Please run 'pip install uvicorn' and retry.")
        sys.exit(1)
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8443,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
