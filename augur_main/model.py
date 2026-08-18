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
        y=wav, n_fft=1024, hop_length=int(sr / 128), fmin=1000, fmax=10000
    )
    mels = librosa.power_to_db(mels)
    mels -= np.mean(mels)
    mels /= np.std(mels)
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

        # model architecture
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(7, 7),
                padding="same",
            ),
            nn.GroupNorm(8, 32),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(5, 5),
                padding="same",
            ),
            nn.GroupNorm(16, 64),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                padding="same",
            ),
            nn.GroupNorm(16, 128),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(3, 3),
                padding="same",
                groups=128,
            ),
            nn.GroupNorm(16, 128),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(3, 3),
                padding="same",
                groups=128,
                dilation=2,
            ),
            nn.GroupNorm(16, 128),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(1, 1),
                padding="same",
            ),
            nn.GroupNorm(32, 256),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(3, 3),
                padding="same",
                groups=256,
            ),
            nn.GroupNorm(32, 256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(3, 3),
                padding="same",
                groups=256,
                dilation=2,
            ),
            nn.GroupNorm(32, 256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=(1, 1),
                padding="same",
            ),
            nn.GroupNorm(64, 512),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                padding="same",
                groups=512,
            ),
            nn.GroupNorm(64, 512),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                padding="same",
                groups=512,
                dilation=2,
            ),
            nn.GroupNorm(64, 512),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=1024,
                kernel_size=(1, 1),
                padding="same",
            ),
            nn.GroupNorm(128, 1024),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1),
        )

    # feeds a batch of 128 by 129 pixel mel spectrograms into the model
    def forward(self, mels):
        mels = torch.reshape(mels, (mels.shape[0], 1, 128, 129))
        return torch.reshape(self.model(mels), (-1,))

    def classify(
        self,
        audio,
        threshold=0.5,
        return_probs=True,
        sample_rate=22050,
    ):
        assert (
            len(audio) >= sample_rate
        ), "Cannot classify audio segments less than 1s..."
        has_song = False
        probs = np.zeros(len(audio))
        if len(audio) != sample_rate:
            seconds = (len(audio) // sample_rate) + 1
            audio = librosa.util.fix_length(audio, size=seconds * sample_rate)
        else:
            seconds = 1
        windows = seconds * 2 - (1)
        with torch.no_grad():
            for i in range(windows):
                window = audio[(i * sample_rate) // 2 : ((i + 2) * sample_rate) // 2]
                mels = generate_spectrogram(window, sr=sample_rate)
                mels = torch.unsqueeze(mels, dim=0).to(torch.float32)
                logit = self.forward(mels).item()
                prob = 1 / (1 + np.exp(-logit))
                if prob >= threshold:
                    has_song = True
                    if not return_probs:
                        return has_song
                if len(audio) > sample_rate:
                    window = probs[
                        (i * sample_rate) // 2 : ((i + 2) * sample_rate) // 2
                    ]
                    window[:] = prob
                else:
                    probs[:] = prob
        if return_probs:
            return has_song, probs
        return has_song
