# train_hkl_vits.py

model = HKLVITS(
    vocab_size=150,
    phoneme_vocab=80
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=2e-4
)

for batch in dataloader:

    text = batch["text"]
    phonemes = batch["phonemes"]
    pitch = batch["pitch"]
    energy = batch["energy"]
    audio = batch["audio"]

    pred_audio = model(text,phonemes,pitch,energy)

    loss = compute_loss(pred_audio,audio)

    loss.backward()

    optimizer.step()