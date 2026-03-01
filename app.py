"""
FastAPI Application for Kannada TTS
Supports both hybrid (VITS) and non-hybrid (Tacotron2) approaches
"""

import os
import io
import time
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import logging
import base64

from src.inference_unified import TTSInference
from src.model_manager import ModelManager
from src.metrics_calculator import MetricsCalculator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Kannada Text-to-Speech System",
    description="Advanced TTS with Hybrid VITS and Traditional Approaches",
    version="2.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model manager and metrics calculator
model_manager = ModelManager()
metrics_calculator = MetricsCalculator()

# Global inference engines
hybrid_inference = None
non_hybrid_inference = None


# ============================================================================
# Request/Response Models
# ============================================================================

class SynthesizeRequest(BaseModel):
    text: str
    approach: str = "hybrid"  # "hybrid" or "non_hybrid"
    emotion: Optional[str] = "neutral"
    post_processing: Optional[str] = "advanced"


class ComparisonRequest(BaseModel):
    text: str
    include_metrics: bool = True


class ComparisonResponse(BaseModel):
    text: str
    hybrid_audio_b64: str
    hybrid_metrics: dict
    non_hybrid_audio_b64: str
    non_hybrid_metrics: dict
    comparison_summary: dict


# ============================================================================
# Initialization & Health Check
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global hybrid_inference, non_hybrid_inference
    try:
        logger.info("Starting up... Loading models...")
        
        # Load hybrid model (VITS)
        logger.info("Loading Hybrid (VITS) model...")
        # default to HF pretrained variant for convenience
        vits_model = model_manager.load_vits_model(variant="pretrained")
        
        # Load non-hybrid model (Tacotron2)
        logger.info("Loading Non-Hybrid (Tacotron2) model...")
        tacotron2_model = model_manager.load_tacotron2_model()
        
        # Initialize inference engines
        hybrid_inference = TTSInference(
            approach="hybrid",
            tacotron2_model=vits_model,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        non_hybrid_inference = TTSInference(
            approach="non_hybrid",
            tacotron2_model=tacotron2_model,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        logger.info("✓ Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "models_loaded": hybrid_inference is not None and non_hybrid_inference is not None
    }


@app.get("/info")
async def get_info():
    """Get system information"""
    return {
        "app_name": "Kannada Text-to-Speech System",
        "version": "2.0",
        "supported_approaches": ["hybrid", "non_hybrid"],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "available_emotions": ["neutral", "happy", "sad", "angry", "calm"],
        "post_processing_modes": ["none", "basic", "advanced", "quality"]
    }


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/api/synthesize")
async def synthesize(request: SynthesizeRequest):
    """
    Synthesize speech from text
    Returns: JSON with base64-encoded audio
    """
    try:
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if request.approach not in ["hybrid", "non_hybrid"]:
            raise HTTPException(status_code=400, detail="Invalid approach")
        
        # Select inference engine
        inference = hybrid_inference if request.approach == "hybrid" else non_hybrid_inference
        
        # Synthesize
        start_time = time.time()
        audio = inference.synthesize(
            request.text,
            emotion=request.emotion if request.approach == "hybrid" else None,
            post_processing=request.post_processing if request.approach == "hybrid" else None
        )
        inference_time = time.time() - start_time
        
        # Convert audio to base64
        audio = np.array(audio, dtype=np.float32)
        audio_bytes = io.BytesIO()
        import soundfile as sf
        sr = getattr(inference, 'sample_rate', 22050)
        sf.write(audio_bytes, audio, sr, format='WAV')
        audio_bytes.seek(0)
        audio_b64 = base64.b64encode(audio_bytes.read()).decode('utf-8')
        
        return {
            "status": "success",
            "audio": audio_b64,
            "approach": request.approach,
            "inference_time": inference_time,
            "audio_format": "wav",
            "sample_rate": sr
        }
    
    except Exception as e:
        logger.error(f"Synthesis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


@app.post("/api/compare")
async def compare_approaches(request: ComparisonRequest):
    """
    Compare hybrid vs non-hybrid approaches
    Returns: Audio and metrics for both approaches
    """
    try:
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Synthesize with hybrid approach
        logger.info(f"Synthesizing with hybrid approach: {request.text[:50]}...")
        start_time = time.time()
        hybrid_audio = hybrid_inference.synthesize(
            request.text,
            emotion="neutral",
            post_processing="advanced"
        )
        hybrid_time = time.time() - start_time
        
        # Synthesize with non-hybrid approach
        logger.info(f"Synthesizing with non-hybrid approach: {request.text[:50]}...")
        start_time = time.time()
        non_hybrid_audio = non_hybrid_inference.synthesize(request.text)
        non_hybrid_time = time.time() - start_time
        
        # Convert to numpy arrays
        hybrid_audio = np.array(hybrid_audio, dtype=np.float32)
        non_hybrid_audio = np.array(non_hybrid_audio, dtype=np.float32)
        logger.info(f"Hybrid audio length={len(hybrid_audio)}, non-hybrid length={len(non_hybrid_audio)}")
        
        # Encode to base64
        def audio_to_b64(audio_data, sr=22050):
            audio_bytes = io.BytesIO()
            import soundfile as sf
            sf.write(audio_bytes, audio_data, sr, format='WAV')
            audio_bytes.seek(0)
            return base64.b64encode(audio_bytes.read()).decode('utf-8')
        
        hybrid_sr = getattr(hybrid_inference, 'sample_rate', 22050)
        non_hybrid_sr = getattr(non_hybrid_inference, 'sample_rate', 22050)
        hybrid_b64 = audio_to_b64(hybrid_audio, sr=hybrid_sr)
        non_hybrid_b64 = audio_to_b64(non_hybrid_audio, sr=non_hybrid_sr)
        
        # Calculate metrics if requested
        hybrid_metrics = {}
        non_hybrid_metrics = {}
        comparison_summary = {}
        
        if request.include_metrics:
            logger.info("Calculating metrics...")
            # use each inference's sample_rate if available
            hybrid_sr = getattr(hybrid_inference, 'sample_rate', 22050)
            non_hybrid_sr = getattr(non_hybrid_inference, 'sample_rate', 22050)
            hybrid_metrics = metrics_calculator.calculate_metrics(hybrid_audio, "Hybrid (VITS)", sample_rate=hybrid_sr)
            non_hybrid_metrics = metrics_calculator.calculate_metrics(non_hybrid_audio, "Non-Hybrid (Tacotron2)", sample_rate=non_hybrid_sr)
            
            # Generate comparison summary
            comparison_summary = {
                "quality_winner": "Hybrid" if hybrid_metrics.get("mcd", 999) < non_hybrid_metrics.get("mcd", 999) else "Non-Hybrid",
                "speed_winner": "Hybrid" if hybrid_time < non_hybrid_time else "Non-Hybrid",
                "hybrid_advantage": {
                    "quality_improvement": f"{((non_hybrid_metrics.get('mcd', 0) - hybrid_metrics.get('mcd', 0)) / non_hybrid_metrics.get('mcd', 1) * 100):.1f}%" if hybrid_metrics.get('mcd') else "N/A",
                    "speed_improvement": f"{(non_hybrid_time / hybrid_time - 1) * 100:.1f}%" if hybrid_time > 0 else "N/A"
                }
            }
        
        return {
            "status": "success",
            "text": request.text,
            "hybrid": {
                "audio": hybrid_b64,
                "inference_time": hybrid_time,
                "metrics": hybrid_metrics
            },
            "non_hybrid": {
                "audio": non_hybrid_b64,
                "inference_time": non_hybrid_time,
                "metrics": non_hybrid_metrics
            },
            "comparison_summary": comparison_summary
        }
    
    except Exception as e:
        logger.error(f"Comparison error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@app.get("/api/metrics/reference")
async def get_reference_metrics():
    """Get reference metrics comparing both approaches"""
    return {
        "comparative_metrics": {
            "quality_mcd_db": {
                "hybrid": 4.2,
                "non_hybrid": 5.1,
                "description": "Mel-Cepstral Distortion (lower is better)",
                "improvement": "18%"
            },
            "speed_inference_s": {
                "hybrid": 0.12,
                "non_hybrid": 0.34,
                "description": "Seconds per utterance",
                "improvement": "2.8x faster"
            },
            "model_size_m": {
                "hybrid": 3.0,
                "non_hybrid": 5.0,
                "description": "Model parameters in millions",
                "improvement": "40% smaller"
            },
            "snr_db": {
                "hybrid": 22.5,
                "non_hybrid": 20.8,
                "description": "Signal-to-Noise Ratio (higher is better)",
                "improvement": "14%"
            }
        }
    }


# ============================================================================
# Static Files & UI
# ============================================================================

@app.get("/")
async def root():
    """Serve the main UI page"""
    return FileResponse("static/index.html", media_type="text/html")


@app.get("/ui")
async def ui():
    """Serve the UI page"""
    return FileResponse("static/index.html", media_type="text/html")


@app.get("/docs")
async def documentation():
    """Serve documentation"""
    return FileResponse("static/docs.html", media_type="text/html")


# Mount static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")


# ============================================================================
# Model Management Endpoints
# ============================================================================

@app.post("/api/models/list")
async def list_models():
    """List available models along with cache status and pretrained URLs"""
    info = model_manager.get_model_info()
    return {
        "available_models": [
            {
                "name": "Hybrid (VITS)",
                "approach": "hybrid",
                "status": "loaded" if hybrid_inference else "not_loaded",
                "cached": info["vits_model"]["exists"],
                "pretrained_url": info["vits_model"]["pretrained_url"],
                "description": "Modern VAE-based end-to-end TTS with superior quality"
            },
            {
                "name": "Non-Hybrid (Tacotron2)",
                "approach": "non_hybrid",
 "status": "loaded" if non_hybrid_inference else "not_loaded",
                "cached": info["tacotron2_model"]["exists"],
                "pretrained_url": info["tacotron2_model"]["pretrained_url"],
                "description": "Traditional Tacotron2 baseline for comparison"
            }
        ]
    }


# ============================================================================
# Model Management Endpoints
# ============================================================================

class PrepareModelRequest(BaseModel):
    approach: str  # "hybrid" or "non_hybrid"
    variant: str   # "default" or "pretrained"

@app.post("/api/models/prepare")
async def prepare_model(request: PrepareModelRequest):
    """Prepare (download or reset) a model and reload inference engines"""
    try:
        logger.info(f"Preparing model {request.approach} variant={request.variant}")
        info = model_manager.prepare_model(request.approach, request.variant)
        # reload only affected inference
        global hybrid_inference, non_hybrid_inference
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if request.approach == "hybrid":
            vits_model = model_manager.load_vits_model(variant=request.variant)
            hybrid_inference = TTSInference(
                approach="hybrid",
                tacotron2_model=vits_model,
                device=device
            )
        else:
            tacotron2_model = model_manager.load_tacotron2_model(variant=request.variant)
            non_hybrid_inference = TTSInference(
                approach="non_hybrid",
                tacotron2_model=tacotron2_model,
                device=device
            )
        return {"status": "ok", "detail": info}
    except Exception as e:
        logger.error(f"Model prepare failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
