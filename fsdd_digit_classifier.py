import os
import glob
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

SAMPLE_RATE = 8000
MAX_SECONDS = 1.0
MAX_LEN = int(SAMPLE_RATE * MAX_SECONDS)

### ======== dataset ==========
# loading in wavs
class fsddDataset(Dataset):
    def __init__(self, wav_paths, transform=None):
        self.wav_paths = wav_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.wav_paths)
    
    def __getitem__(self, idx):
        path = self.wav_paths[idx]
        label = int(Path(path).stem.split('_')[0])
        waveform, sr = torchaudio.load(path)

        # Resample to match training sample rate
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Pad/trim to fixed length
        if waveform.shape[1] < MAX_LEN:
            pad = MAX_LEN - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :MAX_LEN]

        # Normalize audio (zero mean, unit variance)
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-6)

        if self.transform:
            feat = self.transform(waveform)
        else:
            feat = waveform

        return feat, label

### ======== cnn model ==========
class smallCNN(nn.Module):
    def __init__(self, in_channels=1, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
             nn.Conv2d(in_channels, 8, kernel_size=3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
            nn.MaxPool2d(2),   # reduce
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(32, n_classes)
        )
    
    def forward(self, x):
        return self.net(x)

# converting waveform to a compact feature
# if doesn't work, try wav to spectrogram
def make_mel_spectrogram_transform(n_fft=256, hop_length=128, n_mels=40):
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    def transform(waveform):
        M = mel(waveform)
        M = torch.log1p(M)  # log scale
        return M
    return transform

# file list
def prepare_file_list(data_dir='data/recordings'):
    # collects .wav files in folder
    files = sorted(glob.glob(os.path.join(data_dir, '*.wav')))
    if not files:
        raise FileNotFoundError(f"No WAVs found in {data_dir}. Put FSDD recordings into that folder.")
    return files

def collate_fn(batch):
    feats = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    # stack (they should be same shape) - ensure shape [B, C, F, T]
    feats = torch.stack(feats)
    return feats, labels

### ======== training loop ==========
def train_one_epoch(model, loader, optim, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optim.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optim.step()
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total

def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total

### ======== training loop ==========
def main(data_dir='data/recordings', epochs=12, batch_size=64, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    files = prepare_file_list(data_dir)
    # quick train/val split: stratify by label to keep distribution
    labels = [int(Path(p).stem.split('_')[0]) for p in files]
    train_files, val_files = train_test_split(files, test_size=0.2, stratify=labels, random_state=42)

    spec_transform = make_mel_spectrogram_transform(n_fft=256, hop_length=128)
    transform = spec_transform

    train_ds = fsddDataset(train_files, transform=transform)
    val_ds = fsddDataset(val_files, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # inspect a sample to get channel dims
    sample_feat, _ = train_ds[0]
    # sample_feat might be (n_mels, time) or (1, n_mels, time)
    if sample_feat.dim() == 2:
        in_ch = 1
        # wrap dataset transform to add channel dim
    else:
        in_ch = sample_feat.shape[0]

    model = smallCNN(in_channels=in_ch, n_classes=10).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for ep in range(1, epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optim, criterion, device)
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)
        print(f"Epoch {ep:02d}  train_loss={train_loss:.4f} acc={train_acc:.3f}  val_loss={val_loss:.4f} acc={val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_fsdd_mel.pth")
    print("Done. Best val acc:", best_acc)

### ======== microphone inference ==========
def predict_from_waveform(model, waveform, transform, device):
    # match training preprocessing
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if waveform.shape[1] < MAX_LEN:
        pad = MAX_LEN - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    else:
        waveform = waveform[:, :MAX_LEN]
    waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-6)
    feat = transform(waveform)
    feat = feat.unsqueeze(0).to(device)  # add batch dim
    model.eval()
    with torch.no_grad():
        logits = model(feat)
        pred = logits.argmax(1).item()
    return pred

import time
import sounddevice as sd
import torch

def record_and_predict(model, transform, device, seconds=2.0):
    import sounddevice as sd
    import time

    print("Get ready to speak a digit...")
    time.sleep(1)
    for i in range(3, 0, -1):
        print(i)
        time.sleep(1)
    print("Speak NOW!")

    # Record longer than needed
    recording = sd.rec(int(seconds * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE,
                       channels=1,
                       dtype='float32')
    sd.wait()
    print("Recording done, processing...")

    waveform = torch.from_numpy(recording.T.copy())

    # Convert to mono and normalize
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Apply voice activity detection to trim silence
    try:
        # VAD expects int16 PCM
        vad_waveform = (waveform * 32768).short()
        trimmed = torchaudio.functional.vad(vad_waveform, SAMPLE_RATE)
        trimmed = trimmed.float() / 32768.0
        waveform = trimmed.unsqueeze(0) if trimmed.dim() == 1 else trimmed
    except Exception as e:
        print("VAD failed, using raw audio:", e)

    # Pad or trim to match training length
    if waveform.shape[1] < MAX_LEN:
        pad = MAX_LEN - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    else:
        waveform = waveform[:, :MAX_LEN]

    # Normalize audio
    waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-6)

    # Predict
    pred = predict_from_waveform(model, waveform, transform, device)
    print("Predicted digit:", pred)
    return pred

### ======== debugging ==========
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np

def debug_pipeline_consistency(
    wav_path,
    training_transform,
    inference_predict_fn,
    model,
    device
):
    # === Load file ===
    waveform, sr = torchaudio.load(wav_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    # --- TRAINING PIPELINE SIMULATION ---
    wf_train = waveform.clone()
    # same normalization as training
    wf_train = (wf_train - wf_train.mean()) / (wf_train.std() + 1e-6)
    spec_train = training_transform(wf_train)
    spec_train = torch.log(spec_train + 1e-9)  # same log step as training

    # --- INFERENCE PIPELINE ---
    # here we hijack predict_from_waveform but intercept before softmax
    wf_inf = waveform.clone()
    wf_inf = (wf_inf - wf_inf.mean()) / (wf_inf.std() + 1e-6)
    spec_inf = training_transform(wf_inf)  # ensure same transform is passed
    spec_inf = torch.log(spec_inf + 1e-9)

    # === Compare ===
    diff = torch.abs(spec_train - spec_inf).max().item()
    print(f"Max absolute difference between pipelines: {diff:.6f}")
    if diff < 1e-6:
        print("Training and inference preprocessing match perfectly!")
    else:
        print("Mismatch detected — predictions may be wrong.")

    # === Plot ===
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(spec_train.squeeze().numpy(), origin='lower', aspect='auto')
    axs[0].set_title("Training pipeline spectrogram")
    axs[1].imshow(spec_inf.squeeze().numpy(), origin='lower', aspect='auto')
    axs[1].set_title("Inference pipeline spectrogram")
    plt.show()

    # === Run prediction ===
    pred_digit = inference_predict_fn(model, waveform, training_transform, device)
    print(f"Model prediction: {pred_digit}")

if __name__ == "__main__":
    # Basic run: assumes dataset in data/recordings
    # main(data_dir='data/recordings', epochs=20, batch_size=64)
    # Example: to run mic inference after training:
    model = smallCNN(in_channels=1, n_classes=10); model.load_state_dict(torch.load('best_fsdd_mel.pth'))
    """debug_pipeline_consistency(
         wav_path="data/recordings/7_theo_49.wav",
         training_transform=make_mel_spectrogram_transform(n_fft=256, hop_length=128, n_mels=40),
         inference_predict_fn=predict_from_waveform,
         model=model,
         device='cuda' if torch.cuda.is_available() else 'cpu'
     )"""
    transform = make_mel_spectrogram_transform(n_mels=40)
    record_and_predict(model, transform, device='cuda' if torch.cuda.is_available() else 'cpu')