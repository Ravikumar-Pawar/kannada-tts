import sys, os, io, time, base64
import numpy as np
sys.path.append(os.getcwd())
from src.model_manager import ModelManager
from src.inference_unified import TTSInference
from src.metrics_calculator import MetricsCalculator

mm = ModelManager()
hybrid_model = mm.load_vits_model(variant='pretrained')
tac = mm.load_tacotron2_model()

hybrid_inf = TTSInference(approach='hybrid', tacotron2_model=hybrid_model)
non_inf = TTSInference(approach='non_hybrid', tacotron2_model=tac)

text = "ನಮಸ್ಕಾರ, ಕನ್ನಡ ವಾಕ್ ಸಂಶ್ಲೇಷಣಾ ವ್ಯವಸ್ಥೆಗೆ ಸ್ವಾಗತ"

print('calling hybrid')
hybrid_audio = hybrid_inf.synthesize(text, emotion='neutral', post_processing='advanced')
print('hybrid_audio', type(hybrid_audio), len(hybrid_audio))
print('calling non-hybrid')
non_audio = non_inf.synthesize(text)
print('non_audio', type(non_audio), len(non_audio))

# conversion
def audio_to_b64(audio_data):
    audio_bytes = io.BytesIO()
    import soundfile as sf
    sf.write(audio_bytes, audio_data, 22050, format='WAV')
    audio_bytes.seek(0)
    return base64.b64encode(audio_bytes.read()).decode('utf-8')

print('converting')
hybrid_b64 = audio_to_b64(np.array(hybrid_audio,dtype=np.float32))
non_b64 = audio_to_b64(np.array(non_audio,dtype=np.float32))
print('converted lengths', len(hybrid_b64), len(non_b64))

# metrics
mc = MetricsCalculator(sample_rate=22050)
print('metrics hybrid', mc.calculate_metrics(np.array(hybrid_audio,dtype=np.float32),'hy'))
print('metrics non', mc.calculate_metrics(np.array(non_audio,dtype=np.float32),'non'))
