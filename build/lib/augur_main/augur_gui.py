import sys
from multiprocessing import Process
import sounddevice as sd
from collections import deque
from screeninfo import get_monitors
from pathlib import Path
import librosa
import os
import traceback
import shutil
import numpy as np
import soundfile as sf
import torch
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QSizePolicy,
    QWidget,
    QGridLayout,
    QPushButton,
    QComboBox,
    QVBoxLayout,
    QFileDialog,
    QLabel,
    QLineEdit,
)
import pyqtgraph as pg

from augur_main.augur_model import AugurModel


def record_and_detect(
    input_device,
    song_dest,
    model_path,
    threshold,
    padding_seconds=5,
    rate=22050,
):
    try:

        # Create audio stream
        stream = sd.InputStream(
            # device=self.device_box.currentData(),
            device=input_device,
            samplerate=rate,
            blocksize=(rate // 2),
        )
        stream.start()

        # Load model
        print("Loading model...")
        model = AugurModel()
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        print("Model loaded!")

        # Create data structure to hold 0.5s segments of audio from the input stream
        chunks = deque()

        # Create two variables before reading audio from the stream. has_song is True iff at least one of the
        # audio segments stored in chunks is classified as song. found_songs stores the number of songs found while
        # the program is running.
        has_song = False
        found_songs = 0

        # Create the counter variable, which counts how many more 0.5s segments of audio the program will read
        # before either saving the audio stored in chunks or removing the leftmost segment stored in chunks.
        counter = padding_seconds * 2

        # Loop that processed audio from the input stream. Runs until stop_button is pressed, deleting the thread.
        while True:
            # Read a 0.5s segment of audio from the input stream
            chunk = stream.read((rate // 2))
            chunks.append(chunk)
            # If the segment contains song, set has_song to true and set counter to padding_seconds * 2.
            if model.classify(chunk, threshold=threshold):
                if not has_song:
                    has_song = True
                    found_songs += 1
                counter = padding_seconds * 2
            # Otherwise, decrease counter by one.
            else:
                counter -= 1
            if counter == 0:
                # If counter equals zero and chunks contains song, save the segments in chunks as an audio file and
                # remove audio segments from chunks until chunks contains only padding_seconds of audio
                if has_song:
                    audio = np.empty(len(chunk) * len(chunks), dtype=np.float32)
                    n = len(chunks)
                    for i in range(0, n):
                        chunk = chunks.popleft()
                        # Ensures that chunks contains padding_seconds of audio after saving audio.
                        if len(chunks) <= padding_seconds * 2:
                            chunks.append(chunk)
                        audio[i * len(chunk) : (i + 1) * len(chunk)] = chunk
                    assert len(chunks) == padding_seconds * 2
                    sf.write(
                        Path(song_dest) / f"found_song_{found_songs}.wav",
                        audio,
                        rate,
                    )
                    out = Path(song_dest) / f"found_song_{found_songs}.wav"
                    print(f"Saved {out}")
                    has_song = False
                    counter = padding_seconds * 2
                # If counter equals zero but chunks does not contain song, remove the leftmost segment in chunks
                # and increment counter by one.
                else:
                    chunks.popleft()
                    counter += 1

    except Exception as e:
        print(e)


def process_folder(
    model,
    input_folder,
    output_folder,
    channel,
    threshold,
    overlap_windows=True,
    clear=False,
):
    if not isinstance(model, AugurModel):
        print("Loading model...")
        m = AugurModel()
        m.load_state_dict(torch.load(model, weights_only=True))
        m.eval()
        model = m
        print("Model loaded!")
    local_output = f"Found Song ({threshold})"
    local_output = Path(input_folder) / local_output
    subdirs = [file for file in Path(input_folder).iterdir() if file.is_dir()]
    if len(subdirs) > 0:
        for subdir in subdirs:
            if not "Found Song" in subdir.name:
                process_folder(model, subdir, output_folder, channel, threshold)
    if any(Path(input_folder).glob("*.wav")):
        if local_output.exists():
            for file in local_output.glob("*.wav"):
                file.unlink()
        local_output.mkdir(parents=True, exist_ok=True)
        if output_folder is not None:
            print(f"outputting to {local_output} and {output_folder}")
        else:
            print(f"outputting to {local_output}")
        for file in Path(input_folder).glob("*.wav"):
            try:
                print(f"processing {file.name}")
                audio, sr = librosa.load(file, sr=22050, mono=False)
                if audio.ndim > 1:
                    audio = audio[channel]
                if model.classify(
                    audio=audio,
                    threshold=threshold,
                    sample_rate=sr,
                    overlap_windows=overlap_windows,
                ):
                    shutil.copy(file, local_output)
                    if output_folder is not None:
                        shutil.copy(
                            file,
                            Path(output_folder) / f"{input_folder.name}_{file.name}",
                        )
                    print(f"found song in {file.name}")
            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                print(f"something went wrong... skipping file {file.name}")
        if clear:
            os.system("cls" if os.name == "nt" else "clear")
        print(f"finished processing {str(input_folder)}!")
    else:
        print(f"{str(input_folder)} contained no audio files")


class AugurGUI(QWidget):
    def __init__(self):
        super().__init__()

        # Set up the window
        self.setWindowTitle(" GUI")
        width, height = get_monitors()[0].width // 4, get_monitors()[0].height // 3
        self.setGeometry(0, 0, width, height)

        # Set the layout for the window
        layout = QGridLayout()
        self.setLayout(layout)

        # Set processes to none until user starts process
        self.recording_process = None
        self.filtering_process = None

        # Set folder paths to None until user chooses
        self.song_loc = None
        self.song_dest = None

        # Create buttons
        self.ofolder_button = QPushButton("Choose output folder", self)
        self.ifolder_button = QPushButton("Choose input folder", self)
        self.start_button = QPushButton("Start recording", self)
        self.stop_button = QPushButton("Stop recording", self)
        self.filter_button = QPushButton("Filter song from existing folder")
        self.start_button.clicked.connect(self._start_recording)
        self.stop_button.clicked.connect(self._stop_recording)
        self.filter_button.clicked.connect(self._filter_song)
        self.ifolder_button.clicked.connect(self._choose_folder)
        self.ofolder_button.clicked.connect(self._choose_folder)
        layout.addWidget(self.ifolder_button, 4, 1)
        layout.addWidget(self.ofolder_button, 5, 1)
        layout.addWidget(self.filter_button, 6, 0, 1, 2)
        font = self.filter_button.font()
        font.setBold(True)
        self.filter_button.setFont(font)
        layout.addWidget(self.start_button, 9, 0)
        layout.addWidget(self.stop_button, 9, 1)

        # Create labels
        self.settings_label = QLabel("Classification settings", self)
        self.channel_label = QLabel("Input channel:", self)
        self.threshold_label = QLabel("Threshold:", self)
        self.folders_label = QLabel("Input/output locations", self)
        self.recording_label = QLabel("Filter song during live recording", self)
        self.ifolder_label = QLabel("No input folder selected:", self)
        self.ofolder_label = QLabel("No output folder selected:", self)
        self.device_label = QLabel("Select input device:", self)
        layout.addWidget(self.settings_label, 0, 0, 1, 2)
        layout.addWidget(self.channel_label, 1, 0)
        layout.addWidget(self.threshold_label, 2, 0)
        layout.addWidget(self.folders_label, 3, 0, 1, 2)
        layout.addWidget(self.ifolder_label, 4, 0)
        layout.addWidget(self.ofolder_label, 5, 0)
        layout.addWidget(self.recording_label, 7, 0, 1, 2)
        layout.addWidget(self.device_label, 8, 0)
        font = self.folders_label.font()
        font.setBold(True)
        self.settings_label.setFont(font)
        self.folders_label.setFont(font)
        self.recording_label.setFont(font)
        self.settings_label.setAlignment(Qt.AlignCenter)
        self.folders_label.setAlignment(Qt.AlignCenter)
        self.recording_label.setAlignment(Qt.AlignCenter)

        # Create input device combobox
        self.device_box = QComboBox()
        for device in sd.query_devices():
            if device["max_input_channels"] > 0:
                self.device_box.addItem(device["name"], device["index"])
        if self.device_box.count() == 0:
            self.device_box.setCurrentText("No input device found...")
            self.start_button.setDisabled(True)
        layout.addWidget(self.device_box, 8, 1)

        # Create text fields
        self.channel_text = QLineEdit()
        self.threshold_text = QLineEdit()
        self.channel_text.setText("0")
        self.threshold_text.setText("0.5")
        layout.addWidget(self.channel_text, 1, 1)
        layout.addWidget(self.threshold_text, 2, 1)

        # Make widgets fill cells
        for i in range(2):
            layout.setColumnStretch(i, 1)

        for widget in [
            self.settings_label,
            self.threshold_label,
            self.channel_label,
            self.folders_label,
            self.start_button,
            self.stop_button,
            self.device_label,
            self.device_box,
            self.filter_button,
            self.ifolder_button,
            self.ifolder_label,
            self.ofolder_button,
            self.ofolder_label,
        ]:
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.channel_text.clearFocus()

    def _start_recording(self):
        if (
            isinstance(self.recording_process, Process)
            and self.recording_process.is_alive()
        ):
            print("Please stop recording before starting again...")
        elif self.song_dest is None:
            print("Please choose a folder to save songs before starting to record")
        else:
            print("Recording started")
            input_device = self.device_box.currentData()
            model_path = Path(__file__).resolve().parent / "model_0.9_0.0731.pt"
            self.recording_process = Process(
                target=record_and_detect,
                args=(
                    input_device,
                    self.song_dest,
                    model_path,
                    float(self.threshold_text.text()),
                ),
            )
            self.recording_process.start()

    def _filter_song(self):
        if (
            isinstance(self.filtering_process, Process)
            and self.filtering_process.is_alive()
        ):
            print(
                "Please wait for filtering process to finish before starting again..."
            )
        if self.song_loc is None:
            print("Please provide an input folder before filtering for song")
        else:
            try:
                model_path = Path(__file__).resolve().parent / "model_0.9_0.0731.pt"
                self.filtering_process = Process(
                    target=process_folder,
                    args=(
                        model_path,
                        self.song_loc,
                        self.song_dest,
                        int(self.channel_text.text()),
                        float(self.threshold_text.text()),
                    ),
                )
                self.filtering_process.start()
            except:
                print(
                    "Please make sure you have correctly specified the input folder..."
                )

    def _stop_recording(self):
        try:
            self.recording_process.terminate()
            self.recording_process = None
            print("Recording ended")
        except AttributeError:
            print("Please start the recording before ending it...")

    def _choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder")
        if folder:
            if self.sender() is self.ifolder_button:
                self.song_loc = folder
                self.ifolder_label.setText(f"Read from: {folder}")
            else:
                self.song_dest = folder
                self.ofolder_label.setText(f"Save to: {folder}")


class RecordingWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Set up the window
        self.setWindowTitle(" GUI")
        width, height = get_monitors()[0].width // 2, get_monitors()[0].height // 3
        layout = QVBoxLayout(self)

        self.plot_widget = pg.PlotWidget()


def main():
    app = QApplication(sys.argv)
    window = AugurGUI()
    window.show()
    app.exec()
    sys.exit()


if __name__ == "__main__":
    main()
