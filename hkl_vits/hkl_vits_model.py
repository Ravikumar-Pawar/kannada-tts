# hkl_vits_model.py

import torch
import torch.nn as nn

from grapheme_encoder import GraphemeEncoder
from phoneme_encoder import PhonemeEncoder
from fusion_layer import FusionLayer
from prosody_encoder import ProsodyEncoder

class HKLVITS(nn.Module):

    def __init__(self, vocab_size, phoneme_vocab):

        super().__init__()

        self.grapheme_encoder = GraphemeEncoder(vocab_size)

        self.phoneme_encoder = PhonemeEncoder(phoneme_vocab)

        self.fusion = FusionLayer()

        self.prosody = ProsodyEncoder()

        # import VITS components
        from TTS.tts.models.vits import VITS

        self.vits = VITS()

    def forward(self, text, phonemes, pitch, energy):

        Hg = self.grapheme_encoder(text)

        Hp = self.phoneme_encoder(phonemes)

        H = self.fusion(Hg,Hp)

        prosody = self.prosody(pitch,energy)

        H = H + prosody

        audio = self.vits(H)

        return audio