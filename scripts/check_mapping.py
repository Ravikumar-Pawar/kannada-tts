import sys, os
sys.path.append(os.path.abspath(os.getcwd()))
from src.hybrid.inference import HybridInference

# create dummy models with minimal interface to avoid .to
class DummyModel:
    def to(self, device):
        return self
    def eval(self):
        return self

dummy = DummyModel()
h = HybridInference(tacotron2_model=dummy, vocoder_model=dummy)
print('mapping size', len(h.character_mapping))
print('ka', 'ಕ' in h.character_mapping)
print('virama', '್' in h.character_mapping)
print('ga', 'ಗ' in h.character_mapping)
print('all keys sample', list(h.character_mapping.keys())[:10])
