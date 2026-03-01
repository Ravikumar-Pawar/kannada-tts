import sys, os
sys.path.append(os.getcwd())
from src.model_manager import ModelManager
from src.hybrid.hf_inference import HFVITSInference
from src.metrics_calculator import MetricsCalculator

mm = ModelManager()
vits = mm.load_vits_model(variant='pretrained')
print('returned type', type(vits))
if isinstance(vits, dict) and vits.get('hf'):
    hf = vits['model']; tok = vits['tokenizer']
    print('hf model loaded, sample rate', getattr(hf.config,'sampling_rate',None))
    inf = HFVITSInference(hf,tok)
    print('Generating audio')
    audio = inf.synthesize('ನಮಸ್ಕಾರ')
    print('audio shape', audio.shape, 'dtype', audio.dtype)
    mc = MetricsCalculator(sample_rate=inf.sample_rate)
    print('metrics', mc.calculate_metrics(audio,'hf'))
else:
    print('unexpected structure')
