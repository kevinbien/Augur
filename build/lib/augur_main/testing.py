import torch
from augur_main.augur_model import SongIdentifier
import librosa

file = "C:\\Users\\kevin\\Downloads\\.wav files only\\T2022-12-16_10-11-26_0001703.wav"
audio, sr = librosa.load(file, sr=None, mono=False)
audio = audio[1]

model = SongIdentifier()
model.load_state_dict(torch.load("C:\\Users\\kevin\\Research\\Augur\\augur_main\\model_2.4_0.991.pt", weights_only=True))
has_song, preds = model.classify(audio, 0.25, sample_rate=sr, overlap_windows=True, numeric_predictions=True)

print(len(preds))
print(preds)
print(has_song)
