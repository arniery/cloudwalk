import torch
import torchaudio
from pathlib import Path
import fsdd_digit_classifier

# Load your trained model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = fsdd_digit_classifier.smallCNN(in_channels=1, n_classes=10).to(device)
model.load_state_dict(torch.load("best_fsdd_mel.pth", map_location=device))

# Use the same transform as training
transform = fsdd_digit_classifier.make_mel_spectrogram_transform(n_fft=256, hop_length=128, n_mels=40)

# Pick a file from your test/validation set
test_wav_path = "data/recordings/5_lucas_21.wav"  # change to one in your dataset

# Load waveform
waveform, sr = torchaudio.load(test_wav_path)

# Resample if needed
if sr != fsdd_digit_classifier.SAMPLE_RATE:
    waveform = torchaudio.functional.resample(waveform, sr, fsdd_digit_classifier.SAMPLE_RATE)

# Predict
pred = fsdd_digit_classifier.predict_from_waveform(model, waveform, transform, device)
print(f"Predicted digit: {pred}")


