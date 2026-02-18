import numpy as np
from pathlib import Path
from datetime import datetime
import soundfile as sf
import torch
from augur_main.model import AugurModel
import sounddevice as sd
from multiprocessing import shared_memory
import librosa
import traceback
import queue


def save_song(audio, song_dest, rate, song_start, song_end):
    if song_start < song_end:
        window = audio[:, song_start * (rate // 2) : song_end * (rate // 2)]
    if song_start > song_end:
        window = audio[0, song_start:]
        window = np.concat((window, audio[0, :song_end]))
        preds = audio[1, song_start:]
        preds = np.concat((preds, audio[1, :song_end]))
        window = np.vstack((window, preds))

    name = f"{str(datetime.now()).replace(':', '-')}.wav"
    sf.write(
        Path(song_dest) / name,
        window.T,
        rate,
    )
    out = Path(song_dest) / name
    print(f"Saved {out}")


q = queue.Queue()

# Taken from https://python-sounddevice.readthedocs.io/en/0.3.14/examples.html
def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())


# padding_seconds: how many seconds of recording before and after a frame containing song will be played
def record_and_detect(
    model_path,
    input_device,
    song_dest,
    shm_name,
    rate,
    max_seconds,
    threshold=0.5,
    padding_seconds=5,
):
    try:
        # Load model
        print("Loading model...")
        model = AugurModel()
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=m.device))
        model.eval()
        threshold = threshold
        print("Model loaded!")

        # Create audio stream
        stream = sd.InputStream(
            device=input_device,
            channels=1,  # Assume mono input is sufficient (allow user to specify this later)
            samplerate=rate,
            blocksize=(rate // 2),
            dtype="float32",
            callback=callback,
        )
        stream.start()

        # Create audio data structure and store in shared memory
        shm = shared_memory.SharedMemory(name=shm_name)
        audio = np.ndarray((2, rate * max_seconds), dtype=np.float32, buffer=shm.buf)
        audio[:, :] = 0

        chunk = 0  # Index of next audio chunk
        song_start = None  # Index of first chunk with song
        chunks_since_song = 0  # Number of chunks since the last chunk with song
        max_chunks = max_seconds * 2  # Maximum number of chunks stored in array

        while True:
            audio[0, chunk * (rate // 2) : (chunk + 1) * (rate // 2)] = np.ravel(q.get())

            if chunk == 0:
                window = np.concat((audio[0, -(rate // 2) :], audio[0, : (rate // 2)]))
            else:
                window = audio[0, (chunk - 1) * (rate // 2) : (chunk + 1) * (rate // 2)]

            has_song, pred = model.classify(
                window, threshold=threshold, sample_rate=rate, numeric_predictions=True
            )

            if chunk == 0:
                audio[1, -(rate // 2) :] = pred
                audio[1, : (rate // 2)] = pred
            else:
                audio[1, (chunk - 1) * (rate // 2) : (chunk + 1) * (rate // 2)] = pred

            if has_song:
                if song_start == None:
                    song_start = (chunk - 2 * padding_seconds) % max_chunks
                chunks_since_song = 0
            else:
                chunks_since_song += 1
            chunk = (chunk + 1) % max_chunks

            if (
                chunks_since_song == padding_seconds * 2
                and song_start != None
                or chunk + 1 == song_start
            ):
                save_song(audio, song_dest, rate, song_start, chunk)
                song_start = None

    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)


# Function for detecting files containing song within a larger folder of recordings and labeling song within them
def process_folder(
    model,
    input_folder,
    output_folder,
    channel,
    threshold,
    overlap_windows,
    excluded,
):
    # Load model when first calling the function
    if not isinstance(model, AugurModel):
        print("Loading model...")
        m = AugurModel()
        m.load_state_dict(torch.load(model, weights_only=True, map_location=m.device))
        m.eval()
        model = m
        print("Model loaded!")

    # Processes subdirectories in the input directory
    subdirs = [file for file in Path(input_folder).iterdir() if file.is_dir()]
    if len(subdirs) > 0:
        for subdir in subdirs:
            process_subdir = True
            for keyword in excluded:
                if keyword in subdir.name:
                    process_subdir = False
            if process_subdir:
                process_folder(
                    model,
                    subdir,
                    output_folder,
                    channel,
                    threshold,
                    overlap_windows,
                    excluded,
                )

    # Processes folders containing .wav files
    if any(Path(input_folder).glob("*.wav")):

        # Creates local "Found Song" folder containing only song-containing files
        local_output = f"Found Song ({threshold}, {str(round((1.0 - 1/overlap_windows) * 100, ndigits=1))}% overlap)"
        local_output = Path(input_folder) / local_output

        # If local_output already exists, remake it
        if local_output.exists():
            for file in local_output.glob("*.wav"):
                file.unlink()
        local_output.mkdir(parents=True, exist_ok=True)

        if output_folder is not None:
            print(f"outputting to {local_output} and {output_folder}")
        else:
            print(f"outputting to {local_output}")

        # Detects song-containing files in input_folder
        for file in Path(input_folder).glob("*.wav"):
            try:
                print(f"processing {file.name}")
                audio, sr = librosa.load(file, sr=22050, mono=False)
                if audio.ndim > 1:
                    input = audio[channel]
                else:
                    input = audio
                has_song, preds = model.classify(
                    audio=input,
                    threshold=threshold,
                    sample_rate=sr,
                    numeric_predictions=True,
                    overlap_windows=overlap_windows,
                )
                if has_song:

                    # Add channel containing labels
                    audio = np.vstack((audio, preds)).T
                    sf.write(
                        local_output / file.name,
                        audio,
                        sr,
                    )
                    if output_folder is not None:
                        sf.write(
                            Path(output_folder) / file.name,
                            audio,
                            sr,
                        )
                    print(f"found song in {file.name}")
            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                print(f"something went wrong... skipping file {file.name}")
        print(f"finished processing {str(input_folder)}!")
