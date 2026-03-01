import sys, os
sys.path.append(os.getcwd())
from src.hybrid.vits_inference import VITSInference
class Dummy:
    def to(self,d):
        return self
    def eval(self):
        return self

model=Dummy()
vi = VITSInference(model)
text = "ನಮಸ್ಕಾರ, ಕನ್ನಡ ವಾಕ್ ಸಂಶ್ಲೇಷಣಾ ವ್ಯವಸ್ಥೆಗೆ ಸ್ವಾಗತ"
print('text:', text)
for ch in text:
    print(ch, hex(ord(ch)), ch in vi.character_mapping)
print('missing', [ch for ch in text if ch not in vi.character_mapping])
