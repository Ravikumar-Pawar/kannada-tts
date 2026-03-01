Terminology and Basic Concepts
==============================

This document explains the key terms and concepts used throughout the Kannada TTS
system. It is intended for readers who are new to text‑to‑speech or neural audio
models.

**Text-to-Speech (TTS)**
:  A process that converts written text into spoken audio. Modern systems use
   machine learning to generate natural-sounding speech.

**Mel-spectrogram**
:  A time‑frequency representation of audio. It shows how energy is distributed
   across perceptually scaled frequency bins (mel scale) over time. Most neural
   TTS models predict mel-spectrograms, which are then converted to waveforms
   using a vocoder or Griffin–Lim algorithm.

**VITS**
:  Variational Inference Text-to-Speech. An end‑to‑end model that combines a
   variational autoencoder (VAE) with a normalizing flow-based generator to
   produce high-quality audio without requiring separate duration models.
   In this project the term ``hybrid`` is often used because the implementation
   includes both the VITS acoustic model and a separate vocoder.

**VAE (Variational Autoencoder)**
:  A generative model that learns to map inputs to a probability distribution
   in a latent space. Training involves minimizing reconstruction loss plus a
   KL divergence term that forces the learned distribution to match a prior.
   VITS uses a VAE for the posterior encoder and a learned prior for sampling
   latent codes during inference.

**KL Divergence**
:  A measure of how one probability distribution differs from another.
   In the VAE training loss, it encourages the posterior distribution (conditioned
   on real audio) to stay close to a standard normal prior. Without this term
   the model may "collapse" and ignore the latent variables.

**Duration Predictor**
:  A network that estimates how long each input token (character/phoneme) should
   be held when generating audio. It is crucial for aligning the text sequence
   to the time dimension of the mel-spectrogram.

**Encoder / Decoder / Generator**
:  Common neural network components.
  * *Encoder* – maps text or audio to latent representations.
  * *Posterior Encoder* – in VITS, processes target mel-spectrograms to produce
    a latent distribution used during training.
  * *Generator* (sometimes called decoder) – takes latent codes (and optionally
    conditioning) and synthesizes mel-spectrograms.

**Prosody**
:  The rhythm, stress, and intonation of speech. Prosody enhancement modules in
   this project adjust pitch, duration, and energy to express emotions such as
   happy, sad, angry, etc.

**Vocoder**
:  A model or algorithm that converts mel-spectrograms into raw audio waveforms.
   This project can use a HiFiGAN-based vocoder (see \`config/hifigan.json\`)
   or the built-in generator for end-to-end inference.

**Inference**
:  The process of running the trained model to generate audio from new text.
   The `VITSInference` class handles tokenization, duration prediction, sampling
   from the latent prior, and optional post‑processing.

**Training**
:  The process of updating model parameters using labeled data (text-audio
   pairs). The `VITSTrainer` class wraps the training loop, loss computation, and
   checkpointing.

**Configuration**
:  JSON files under `config/` that specify hyperparameters for models and
   training. Refer to the configuration guide for meanings and recommended values.

**CLI**
:  Command‑line interface provided by `run_tts.py` that allows quick experiments
   with training, inference, and comparison modes without writing Python code.
