# kannada_g2p.py

import re
import torch
from typing import List, Tuple

class KannadaG2P:
    """
    Grapheme to Phoneme converter for Kannada
    Implements rule-based phoneme conversion for Kannada text
    
    Kannada phoneme inventory includes:
    - Consonants: ka, kha, ga, gha, cha, cha, ja, jha, tta, ttha, da, dha, ta, tha, da, dha, 
                  na, pa, pha, ba, bha, ma, ya, ra, la, va, sha, ssa, sa, ha
    - Vowels: a, aa, i, ii, u, uu, e, ee, o, oo, ai, au
    - Special: anusvara (M), visarga (H), halant (/)
    """
    
    def __init__(self):
        # Kannada vowels (standalone)
        self.vowels = {
            'ಅ': 'a',
            'ಆ': 'aa',
            'ಇ': 'i',
            'ಈ': 'ii',
            'ಉ': 'u',
            'ಊ': 'uu',
            'ಋ': 'ru',
            'ೃ': 'ru',
            'ಎ': 'e',
            'ಏ': 'ee',
            'ಐ': 'ai',
            'ಒ': 'o',
            'ಓ': 'oo',
            'ಔ': 'au',
            'ಌ': 'lu'
        }
        
        # Kannada consonants
        self.consonants = {
            'ಕ': 'ka',
            'ಖ': 'kha',
            'ಗ': 'ga',
            'ಘ': 'gha',
            'ಙ': 'na',
            'ಚ': 'cha',
            'ಛ': 'cha',
            'ಜ': 'ja',
            'ಝ': 'jha',
            'ಞ': 'nya',
            'ಟ': 'tta',
            'ಠ': 'ttha',
            'ಡ': 'da',
            'ಢ': 'dha',
            'ಣ': 'na',
            'ತ': 'ta',
            'ಥ': 'tha',
            'ದ': 'da',
            'ಧ': 'dha',
            'ನ': 'na',
            'ಪ': 'pa',
            'ಫ': 'pha',
            'ಬ': 'ba',
            'ಭ': 'bha',
            'ಮ': 'ma',
            'ಯ': 'ya',
            'ರ': 'ra',
            'ಱ': 'ra',
            'ಲ': 'la',
            'ಳ': 'la',
            'ವ': 'va',
            'ಶ': 'sha',
            'ಷ': 'ssa',
            'ಸ': 'sa',
            'ಹ': 'ha',
            'ಀ': 'ri'
        }
        
        # Kannada vowel modifiers (matras)
        self.matras = {
            'ಾ': 'aa',
            'ಿ': 'i',
            'ೀ': 'ii',
            'ುಾ': 'u',
            'ೂ': 'uu',
            'ೃ': 'ru',
            'ೄ': 'ru',
            'ೆ': 'e',
            'ೇ': 'ee',
            'ೈ': 'ai',
            'ೊ': 'o',
            'ೋ': 'oo',
            'ೌ': 'au'
        }
        
        # Special marks
        self.special = {
            'ಂ': 'M',      # Anusvara
            'ಃ': 'H',      # Visarga
            '್': ''        # Halant (virama) - no vowel
        }
        
        self.phoneme_list = self._build_phoneme_list()
        self.phoneme_to_id = {p: i for i, p in enumerate(self.phoneme_list)}
        self.id_to_phoneme = {i: p for p, i in self.phoneme_to_id.items()}

    def _build_phoneme_list(self) -> List[str]:
        """Build complete list of phonemes"""
        phonemes = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']  # Special tokens
        phonemes.extend(sorted(set(self.vowels.values())))
        phonemes.extend(sorted(set(self.consonants.values())))
        phonemes.extend(sorted(set(self.matras.values())))
        phonemes.extend(sorted(set(self.special.values())))
        return phonemes

    def grapheme_to_phoneme(self, text: str) -> List[str]:
        """
        Convert Kannada grapheme text to phoneme sequence
        
        Args:
            text: Kannada text string
        
        Returns:
            List of phonemes
        """
        phonemes = []
        i = 0
        
        while i < len(text):
            char = text[i]
            
            # Check standalone vowels
            if char in self.vowels:
                phonemes.append(self.vowels[char])
                i += 1
            
            # Check consonants
            elif char in self.consonants:
                phoneme = self.consonants[char]
                j = i + 1
                
                # Check for vowel modifier (matra)
                has_matra = False
                if j < len(text) and text[j] in self.matras:
                    matra_phoneme = self.matras[text[j]]
                    if matra_phoneme:  # Not empty
                        phoneme += matra_phoneme[2:] if matra_phoneme.startswith('aa') else matra_phoneme
                    has_matra = True
                    j += 1
                
                # If no matra, add inherent 'a' vowel
                if not has_matra and char not in text[j:j+1] or (j < len(text) and text[j] != '्'):
                    phoneme += 'a'  # Inherent vowel
                
                phonemes.append(phoneme)
                i = j
            
            # Check special marks
            elif char in self.special:
                special_ph = self.special[char]
                if special_ph:  # Don't add empty strings from halant
                    phonemes.append(special_ph)
                i += 1
            
            else:
                i += 1
        
        return phonemes

    def phoneme_to_id_sequence(self, phonemes: List[str]) -> List[int]:
        """Convert phoneme list to IDs"""
        return [self.phoneme_to_id.get(p, self.phoneme_to_id['<UNK>']) for p in phonemes]

    def id_to_phoneme_sequence(self, ids: List[int]) -> List[str]:
        """Convert ID sequence back to phonemes"""
        return [self.id_to_phoneme.get(i, '<UNK>') for i in ids]

    def text_to_phoneme_ids(self, text: str) -> torch.Tensor:
        """
        Convert Kannada text directly to phoneme ID tensor
        
        Args:
            text: Kannada text string
        
        Returns:
            torch.Tensor of phoneme IDs
        """
        phonemes = self.grapheme_to_phoneme(text)
        ids = self.phoneme_to_id_sequence(phonemes)
        return torch.tensor(ids, dtype=torch.long)

    def batch_text_to_phoneme_ids(self, texts: List[str], pad_length: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert batch of texts to padded phoneme ID tensors
        
        Args:
            texts: List of Kannada text strings
            pad_length: Length to pad to (uses max length if not specified)
        
        Returns:
            Tuple of (phoneme_ids, lengths)
            - phoneme_ids: (batch_size, max_seq_len)
            - lengths: (batch_size,) actual lengths
        """
        phoneme_id_sequences = [self.text_to_phoneme_ids(text).tolist() for text in texts]
        
        if pad_length is None:
            pad_length = max(len(seq) for seq in phoneme_id_sequences)
        
        batch_size = len(phoneme_id_sequences)
        padded = torch.zeros(batch_size, pad_length, dtype=torch.long)
        lengths = torch.zeros(batch_size, dtype=torch.long)
        
        for i, seq in enumerate(phoneme_id_sequences):
            seq_len = min(len(seq), pad_length)
            padded[i, :seq_len] = torch.tensor(seq[:seq_len])
            lengths[i] = seq_len
        
        return padded, lengths

    @property
    def vocab_size(self) -> int:
        """Get phoneme vocabulary size"""
        return len(self.phoneme_list)
