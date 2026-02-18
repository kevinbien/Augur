import numpy as np
import torch
from pathlib import Path
import librosa
from torch import nn


# preprocessing functions
def generate_spectrogram(wav, sr=22050):
    if isinstance(wav, str) or isinstance(wav, Path):
        wav, _ = librosa.load(wav)
    mels = librosa.feature.melspectrogram(
        y=wav, n_fft=1024, hop_length=int(sr / 128), fmin=300, fmax=10000
    )
    mels = librosa.power_to_db(mels)
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

class AugurModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.to(self.device)
        print(f"Using {self.device}")

        # model architecture
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(9, 9),
                padding="same",
            ),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((4, 2), (4, 2)),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(7, 7),
                padding="same",
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(5, 5),
                padding="same",
            ),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 7),
                padding="same",
                dilation=1,
            ),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 7),
                dilation=2,
                padding="same",
            ),
            nn.GroupNorm(8, 64),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=(3, 3),
                padding="same",
            ),
            nn.GroupNorm(8, 32),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(32 * 8 * 16, 1),
        )

    # feeds a batch of 128 by 129 pixel mel spectrograms into the model
    def forward(self, mels):
        mels = torch.reshape(mels, (mels.shape[0], 1, 128, 129))
        return torch.reshape(self.model(mels), (-1,))

    def classify(
        self,
        audio,
        threshold=0.5,
        numeric_predictions=False,
        sample_rate=22050,
        overlap_windows=2,
    ):
        assert (
            len(audio) >= sample_rate
        ), "Cannot classify audio segments less than 1s..."
        has_song = False
        preds = np.zeros(len(audio))
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
            mels = torch.unsqueeze(mels, dim=0).to(torch.float32)
            pred = 1 / (1 + np.exp(-self.forward(mels).item()))
            if pred >= threshold:
                has_song = True
                if not numeric_predictions:
                    return has_song
            if len(audio) > sample_rate:
                preds[
                    (i * sample_rate)
                    // overlap_windows : ((i + overlap_windows) * sample_rate)
                    // overlap_windows
                ] = pred
            else:
                preds = pred
        if numeric_predictions:
            return has_song, preds
        return has_song
