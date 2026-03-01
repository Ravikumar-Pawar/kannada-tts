import os
import shutil
from pathlib import Path
import sys
import types

# ensure a minimal torch stub exists so tests run without installing heavy deps
if 'torch' not in sys.modules:
    torch_stub = types.ModuleType('torch')
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda : False)
    sys.modules['torch'] = torch_stub

import pytest

from src.model_manager import ModelManager


class DummyModel:
    def __init__(self, *args, **kwargs):
        self._state = {}

    def to(self, device):
        return self


def dummy_from_pretrained(repo_id, cache_dir=None, local_files_only=False, **kwargs):
    # simulate caching by creating a marker file
    cache_dir = Path(cache_dir) / repo_id.replace('/', '_')
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "downloaded.txt").write_text("ok")
    return DummyModel()


def dummy_tokenizer(repo_id, cache_dir=None, local_files_only=False, **kwargs):
    return object()


@pytest.fixture(autouse=True)
def cleanup_cache(tmp_path, monkeypatch):
    """Ensure each test gets a fresh cache directory."""
    # ensure the model directory is under tmp_path instead of user home
    monkeypatch.setenv("KANNADA_TTS_MODEL_DIR", str(tmp_path / "models"))
    mm = ModelManager()
    # remove any existing directories just in case
    if mm.model_cache_dir.exists():
        shutil.rmtree(mm.model_cache_dir)
    yield


def test_vits_pretrained_download_and_cache(monkeypatch):
    # monkeypatch transformers imports
    import src.model_manager as mm_module
    monkeypatch.setattr(mm_module, 'VitsModel', type('VitsModel', (), {'from_pretrained': staticmethod(dummy_from_pretrained)}))
    monkeypatch.setattr(mm_module, 'AutoTokenizer', type('AutoTokenizer', (), {'from_pretrained': staticmethod(dummy_tokenizer)}))

    manager = ModelManager()
    # first call should trigger download (dummy)
    result = manager.load_vits_model(variant="pretrained")
    assert isinstance(result, dict) and result.get('hf')
    # cache directory should now exist with marker file
    repo_dir = manager.hf_cache_dir / "facebook_mms-tts-kan"
    assert repo_dir.exists()
    assert (repo_dir / "downloaded.txt").exists()
    # the model folder should be inside the project-local path
    assert str(manager.model_cache_dir).endswith("models")

    # If we call again the same process, the in-memory cache is reused
    result2 = manager.load_vits_model(variant="pretrained")
    assert result2 is result


def test_prepare_model_hybrid(monkeypatch):
    # verify that prepare_model('hybrid','pretrained') populates cache
    import src.model_manager as mm_module
    monkeypatch.setattr(mm_module, 'VitsModel', type('VitsModel', (), {'from_pretrained': staticmethod(dummy_from_pretrained)}))
    monkeypatch.setattr(mm_module, 'AutoTokenizer', type('AutoTokenizer', (), {'from_pretrained': staticmethod(dummy_tokenizer)}))

    manager = ModelManager()
    info = manager.prepare_model('hybrid', 'pretrained')
    assert info.get('status') == 'pretrained_ready'
    assert manager.hf_cache_dir.exists()
