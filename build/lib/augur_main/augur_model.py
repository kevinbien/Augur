import math

import h5py
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import librosa
from torch import nn
from torch.utils.data import Dataset, DataLoader


# preprocessing functions
def generate_spectrogram(wav, sr=22050, mu=-33.4126, sigma=11.2896):
    if isinstance(wav, str) or isinstance(wav, Path):
        wav, _ = librosa.load(wav)
    mels = librosa.feature.melspectrogram(
        y=wav,
        n_fft=1024,
        hop_length=int(sr / 128),
        fmin=300,
        fmax=10000,
    )
    mels = librosa.power_to_db(mels)
    mels -= mu
    mels /= sigma
    mels = torch.from_numpy(mels)
    return mels


class AugurModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.to(self.device)
        print(f"Using {self.device}")
        self.temp = 1.0

        # model architecture
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(9, 9),
                padding="same",
            ),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((4, 4), (4, 4)),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(7, 7),
                padding="same",
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(5, 5),
                padding="same",
            ),
            nn.GroupNorm(8, 64),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.LeakyReLU(),
            nn.Dropout2d(0.05),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                padding="same",
                groups=64,
            ),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(1, 1),
                padding="same",
            ),
            nn.Dropout2d(0.05),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(1, 5),
                padding="same",
                groups=64,
            ),
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=(1, 1),
                padding="same",
            ),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Dropout(0.10),
            nn.Linear(32 * 16 * 16, 1),
        )

        self.num_params = sum(p.numel() for p in self.parameters())

    # feeds a batch of 128 by 129 pixel mel spectrograms into the model
    def forward(self, mels):
        mels = torch.reshape(mels, (mels.shape[0], 1, 128, 129))
        return torch.reshape(self.model(mels), (-1,))

    def classify(
        self,
        audio,
        threshold=0.5,
        numeric_predictions=False,
        print_predictions=False,
        sample_rate=22050,
        overlap_windows=2,
    ):
        assert (
            len(audio) >= sample_rate
        ), "Cannot classify audio segments less than 1s..."
        has_song = False
        preds = []
        if len(audio) != sample_rate:
            seconds = (len(audio) // sample_rate) + 1
            audio = librosa.util.fix_length(audio, size=seconds * sample_rate)
        else:
            seconds = 1
        windows = seconds * overlap_windows - (overlap_windows - 1)
        for i in range(windows):
            window = audio[
                (i * sample_rate)
                // overlap_windows : ((i + overlap_windows) * sample_rate)
                // overlap_windows
            ]
            mels = generate_spectrogram(window, sr=sample_rate)
            mels = torch.unsqueeze(mels, dim=0)
            pred = 1 / (1 + np.exp(-self.forward(mels).item()))
            preds.append(pred)
            if print_predictions:
                print(pred)
            if pred >= threshold:
                has_song = True
                if not numeric_predictions:
                    return has_song
        if numeric_predictions:
            return has_song, preds
        return has_song


class AugurDataset(Dataset):
    def __init__(self, annotations_file, h5py_dataset=None):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.labels = pd.read_csv(annotations_file)
        self.h5py_dataset = h5py_dataset

    def __len__(self):
        return len(self.labels)

    # returns 128 by 129 pixel mel spectrogram of training audio and its corresponding label
    def __getitem__(self, index):
        path = self.labels.iloc[index, 0]
        label = self.labels.iloc[index, 1]
        if self.h5py_dataset is not None:
            mels = torch.from_numpy(np.array(self.h5py_dataset[path])).to(self.device)
        else:
            mels = generate_spectrogram(path)
        return mels, label


def train_loop(dataloader, optimizer, model, loss_fn):
    model.train(True)
    running_loss = 0
    for batch, (data, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        preds = model(data)
        targets = targets.to(torch.float32)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / (batch + 1)
    return avg_loss


def eval_loop(val_dataloader, model, loss_fn):
    model.eval()
    loss_arr = np.zeros(len(val_dataloader))
    with torch.no_grad():
        for batch, (data, targets) in enumerate(val_dataloader):
            preds = model(data)
            targets = targets.to(torch.float32)
            loss_arr[batch] = loss_fn(preds, targets).item()
    return np.mean(loss_arr), np.std(loss_arr) / math.sqrt(len(loss_arr))


def train_model(
    model, dataset, model_dest=None, return_loss=True, batch_size=128, epochs=15
):
    loss_fn = nn.BCELoss()
    best_vloss = 1000000000
    best_vloss_sem = 0
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=True
    )
    for epoch in range(epochs):
        print(f"\n  Epoch {epoch + 1}\n  -------------------------------")
        avg_loss = train_loop(train_dataloader, optimizer, model, loss_fn)
        avg_vloss, vloss_sem = eval_loop(val_dataloader, model, loss_fn)
        scheduler.step(avg_vloss)
        print(f"  LOSS train {avg_loss} valid {avg_vloss}")
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            best_vloss_sem = vloss_sem
            if model_dest is not None:
                model_path = model_dest + f"model_{best_vloss}_{epoch + 1}.pth"
                torch.save(model.state_dict(), model_path)
    if return_loss:
        return best_vloss, best_vloss_sem
