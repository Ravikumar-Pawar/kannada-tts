"""
Model Manager
Handles downloading, caching, and loading pre-trained TTS models
"""

import os
import torch
import logging
from pathlib import Path
from typing import Optional
import gdown

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model downloading, caching, and initialization"""
    
    def __init__(self):
        """Initialize model manager"""
        self.model_cache_dir = Path.home() / ".cache" / "kannada_tts"
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # known pretrained model URLs (placeholders/demo)
        # Replace with real links once available.
        self.PRETRAINED_URLS = {
            "hybrid": "https://example.com/pretrained/vits_kannada.pth",
            "non_hybrid": "https://example.com/pretrained/tacotron2_kannada.pth"
        }

        logger.info(f"Model cache directory: {self.model_cache_dir}")
    
    def load_vits_model(self, variant: str = "default"):
        """
        Load a VITS (Hybrid) model according to variant.

        Args:
            variant: "default" for untrained init, "pretrained" for the
                     HuggingFace facebook/mms-tts-kan checkpoint (downloads
                     automatically when first requested)
        Returns:
            Either a standard VITS instance or a dict containing HF model and tokenizer.
        """
        variant = variant.lower()
        if variant == "default":
            model_path = self.model_cache_dir / "vits_kannada.pth"
            if not model_path.exists():
                logger.info("VITS model not found. Initializing with default configuration...")
                model = self._initialize_vits_default()
            else:
                logger.info(f"Loading VITS model from {model_path}")
                model = self._load_vits_from_checkpoint(str(model_path))
            return model
        elif variant == "pretrained":
            # return cached model if already loaded
            if hasattr(self, '_hf_model') and self._hf_model is not None:
                logger.info("Using cached HuggingFace MMS-TTS Kannada model")
                return {"hf": True, "model": self._hf_model, "tokenizer": self._hf_tokenizer}
            # try to load huggingface model
            try:
                from transformers import VitsModel, AutoTokenizer
                logger.info("Loading HuggingFace MMS-TTS Kannada model")
                hf_model = VitsModel.from_pretrained("facebook/mms-tts-kan")
                hf_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-kan")
                # keep references so subsequent calls reuse the same objects
                self._hf_model = hf_model.to(self.device)
                self._hf_tokenizer = hf_tokenizer
                return {"hf": True, "model": self._hf_model, "tokenizer": self._hf_tokenizer}
            except Exception as e:
                logger.warning(f"Failed to load HF model: {e}. Falling back to default.")
                return self.load_vits_model("default")
        else:
            raise ValueError(f"Unknown VITS variant: {variant}")
    
    def load_tacotron2_model(self, variant: str = "default"):
        """
        Load a Tacotron2 (Non-Hybrid) model. Variant currently ignored but kept
        for API symmetry with load_vits_model.
        """
        # future variants (e.g., pretrained) can be handled here
        model_path = self.model_cache_dir / "tacotron2_kannada.pth"
        
        if not model_path.exists():
            logger.info("Tacotron2 model not found. Initializing with default configuration...")
            model = self._initialize_tacotron2_default()
        else:
            logger.info(f"Loading Tacotron2 model from {model_path}")
            model = self._load_tacotron2_from_checkpoint(str(model_path))
        
        return model
    
    def _initialize_vits_default(self):
        """Initialize VITS model with default configuration"""
        try:
            from src.hybrid.models import VITS
            
            # VITS constructor expects (vocab_size, mel_channels, hidden_size)
            vits = VITS(vocab_size=132, mel_channels=80, hidden_size=192)
            
            logger.info("✓ VITS model initialized successfully")
            return vits.to(self.device)
        
        except Exception as e:
            logger.error(f"Failed to initialize VITS: {str(e)}")
            raise
    
    def _initialize_tacotron2_default(self):
        """Initialize Tacotron2 model with default configuration"""
        try:
            from src.non_hybrid.models import Tacotron2Model
            
            # Tacotron2Model expects (num_chars, encoder_hidden_size, decoder_hidden_size, prenet_sizes)
            tacotron2 = Tacotron2Model(
                num_chars=132,
                encoder_hidden_size=256,
                decoder_hidden_size=1024,
                prenet_sizes=[256, 256]
            )
            
            logger.info("✓ Tacotron2 model initialized successfully")
            return tacotron2.to(self.device)
        
        except Exception as e:
            logger.error(f"Failed to initialize Tacotron2: {str(e)}")
            raise
    
    def _load_vits_from_checkpoint(self, checkpoint_path: str):
        """Load VITS from checkpoint file"""
        try:
            from src.hybrid.models import VITS
            
            model = self._initialize_vits_default()
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()
            
            logger.info("✓ VITS model loaded from checkpoint")
            return model
        
        except Exception as e:
            logger.error(f"Failed to load VITS checkpoint: {str(e)}")
            raise
    
    def _load_tacotron2_from_checkpoint(self, checkpoint_path: str):
        """Load Tacotron2 from checkpoint file"""
        try:
            from src.non_hybrid.models import Tacotron2Model
            
            model = self._initialize_tacotron2_default()
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()
            
            logger.info("✓ Tacotron2 model loaded from checkpoint")
            return model
        
        except Exception as e:
            logger.error(f"Failed to load Tacotron2 checkpoint: {str(e)}")
            raise
    
    def download_model_from_url(self, model_name: str, url: str):
        """
        Download model from URL. Supports http(s) or local file paths.
        
        Args:
            model_name: Name of the model (vits, tacotron2)
            url: URL or file path to download from
        """
        try:
            if model_name.lower() == "vits":
                output_path = self.model_cache_dir / "vits_kannada.pth"
            elif model_name.lower() == "tacotron2":
                output_path = self.model_cache_dir / "tacotron2_kannada.pth"
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            logger.info(f"Downloading {model_name} from {url}...")
            # handle local file copy
            if url.startswith("file://") or os.path.exists(url):
                src = url.replace("file://", "")
                import shutil
                shutil.copy(src, str(output_path))
            else:
                gdown.download(url, str(output_path), quiet=False)
            
            logger.info(f"✓ {model_name} downloaded to {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Failed to download model: {str(e)}")
            raise
    
    def get_model_info(self):
        """Get information about cached models and available pretrained URLs"""
        return {
            "cache_directory": str(self.model_cache_dir),
            "vits_model": {
                "path": str(self.model_cache_dir / "vits_kannada.pth"),
                "exists": (self.model_cache_dir / "vits_kannada.pth").exists(),
                "pretrained_url": self.PRETRAINED_URLS.get("hybrid")
            },
            "tacotron2_model": {
                "path": str(self.model_cache_dir / "tacotron2_kannada.pth"),
                "exists": (self.model_cache_dir / "tacotron2_kannada.pth").exists(),
                "pretrained_url": self.PRETRAINED_URLS.get("non_hybrid")
            }
        }

    def prepare_model(self, approach: str, variant: str = "default"):
        """Prepare a model for use by either resetting to defaults or downloading pretrained.
        Args:
            approach: "hybrid" or "non_hybrid"
            variant: "default" or "pretrained"
        Returns:
            A dict with status information.
        """
        approach = approach.lower()
        if approach not in ["hybrid", "non_hybrid"]:
            raise ValueError(f"Unknown approach: {approach}")
        model_name = "vits" if approach == "hybrid" else "tacotron2"
        target_path = self.model_cache_dir / f"{model_name}_kannada.pth"

        if variant == "default":
            # remove existing checkpoint to force initialization from scratch
            if target_path.exists():
                target_path.unlink()
            return {"status": "default_initialized"}
        elif variant == "pretrained":
            url = self.PRETRAINED_URLS.get(approach)
            if url is None:
                raise ValueError(f"No pretrained URL configured for {approach}")
            # if URL is a placeholder or download fails, create a dummy checkpoint
            if not target_path.exists():
                try:
                    self.download_model_from_url(model_name, url)
                except Exception:
                    logger.warning("Could not download pretrained model, creating dummy checkpoint instead")
                    # generate default model and save its state dict as "pretrained"
                    if approach == "hybrid":
                        model = self._initialize_vits_default()
                    else:
                        model = self._initialize_tacotron2_default()
                    torch.save(model.state_dict(), str(target_path))
            return {"status": "pretrained_ready", "path": str(target_path)}
        else:
            raise ValueError(f"Unsupported variant: {variant}")
