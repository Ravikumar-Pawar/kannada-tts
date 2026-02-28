#!/usr/bin/env python3
from TTS.trainer import Trainer, TrainingArgs
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.datasets import load_tts_samples

config = Tacotron2Config()
config.load_json("config/tacotron2.json")

train_samples, eval_samples = load_tts_samples(
    dataset_config={"formatter": "ljspeech", "meta_file_train": "data/metadata.csv"},
    eval_split_size=0.05
)

model = Tacotron2.init_from_config(config)
trainer = Trainer(
    TrainingArgs(), config, "output/tacotron2/",
    model, train_samples, eval_samples
)
trainer.fit()
