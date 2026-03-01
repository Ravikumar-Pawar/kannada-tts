import sys, os
sys.path.append(os.getcwd())
from src.model_manager import ModelManager

mm = ModelManager()
print('first load')
v1 = mm.load_vits_model(variant='pretrained')
print('done first')
print('second load (should use cache)')
v2 = mm.load_vits_model(variant='pretrained')
print('done second')
print('same dict instance?', v1 is v2)
print('same HF model object?', v1.get('model') is v2.get('model'))
