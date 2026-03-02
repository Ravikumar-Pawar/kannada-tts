"""Utility routines shared by various components of the TTS system.

This module currently contains helpers for Kannada character mapping and
Unicode normalization.  Extracting the logic into a central location
prevents drift between the hybrid (VITS) and non-hybrid code paths.
"""

import unicodedata


def normalize_kannada_text(text: str) -> str:
    """Return the input text normalised to Unicode NFC form.

    Kannada comprises a number of combining characters and diacritics that
    may be encoded in several equivalent ways.  Normalizing to NFC before
    tokenisation ensures the same sequence of characters is produced
    regardless of the original encoding, which in turn makes model behaviour
    deterministic.
    """
    return unicodedata.normalize('NFC', text)


def default_kannada_mapping() -> dict:
    """Build a default character->index mapping for the Kannada script.

    The mapping covers the entire Unicode block U+0C80..U+0CFF and appends a
    small set of punctuation/special symbols that are used throughout the
    dataset and web interface.  Both the hybrid (VITS) and non-hybrid
    inference engines rely on this dictionary by default, guaranteeing
    consistent behaviour and eliminating "character not in mapping"
    warnings for normal Kannada text.
    """
    mapping = {}
    # register every code point in the Kannada block
    for code in range(0x0C80, 0x0CFF + 1):
        ch = chr(code)
        mapping[ch] = len(mapping)
    # extra symbols often seen in text input
    for ch in [' ', '-', '?', '.', ',', '!', ':', ';', '(', ')', '[', ']', '|']:
        if ch not in mapping:
            mapping[ch] = len(mapping)
    return mapping
