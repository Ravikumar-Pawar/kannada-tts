import sys, os
sys.path.append(os.path.abspath(os.getcwd()))
# the hybrid inference class was renamed to VITSInference; we only
# need the mapping helper here
from src.text_utils import default_kannada_mapping

# create dummy models with minimal interface to avoid .to
class DummyModel:
    def to(self, device):
        return self
    def eval(self):
        return self

dummy = DummyModel()
# instead of instantiating an inference object we can call the util directly
mapping = default_kannada_mapping()
print('mapping size', len(mapping))
print('ka', 'ಕ' in mapping)
print('virama', '್' in mapping)
print('ga', 'ಗ' in mapping)
print('all keys sample', list(mapping.keys())[:10])
