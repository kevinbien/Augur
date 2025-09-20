import sys
from multiprocessing import Process
import sounddevice as sd
from collections import deque
from screeninfo import get_monitors
import numpy as np
import soundfile as sf
from importlib import resources
import assets
import torch
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QComboBox
import pyqtgraph

from song_identifier_model import SongIdentifier


def record_and_detect(input_device, song_dest, padding_seconds=5, rate=22050):
    try:

        # Create audio stream
        stream = sd.InputStream(
            # device=self.device_box.currentData(),
            device=input_device,
            samplerate=rate,
            blocksize=(rate // 2),
        )
        stream.start()

        # Create data structure to hold 0.5s segments of audio from the input stream
        chunks = deque()

        # Load model
        print("loading model...")
        model = SongIdentifier()
        model_path = ""
        for file in resources.files(assets).iterdir():
            if ".pth" in str(file):
                model_path = file
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        print("model loaded!")

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
            if model.classify(chunk):
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
                        song_dest + "\\found_song_" + str(found_songs) + ".wav",
                        audio,
                        rate,
                    )
                    print(
                        "saving file "
                        + song_dest
                        + "\\found_song_"
                        + str(found_songs)
                        + ".wav"
                    )
                    has_song = False
                    counter = padding_seconds * 2
                # If counter equals zero but chunks does not contain song, remove the leftmost segment in chunks
                # and increment counter by one.
                else:
                    chunks.popleft()
                    counter += 1

    except Exception as e:
        print(e)


class RealTimeDetector(QWidget):
    def __init__(self):
        super().__init__()

        # Set up the window
        self.setWindowTitle(" GUI")
        width, height = get_monitors()[0].width // 4, get_monitors()[0].height // 3
        self.setGeometry(0, 0, width, height)

        # Set the layout for the window
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create recording process
        self.recording_process = None

        # Create buttons
        self.start_button = QPushButton("Start recording", self)
        self.stop_button = QPushButton("Stop recording", self)
        self.plot_button = QPushButton("Plot audio", self)
        self.start_button.clicked.connect(self._start_recording)
        self.stop_button.clicked.connect(self._stop_recording)
        self.plot_button.clicked.connect(self._plot_audio)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)

        # Create input device combobox
        self.device_box = QComboBox()
        for device in sd.query_devices():
            if device["max_input_channels"] > 0:
                self.device_box.addItem(device["name"], device["index"])
        if self.device_box.count() == 0:
            self.device_box.setCurrentText("No input device found...")
            self.start_button.setDisabled(True)
        layout.addWidget(self.device_box)

    def _start_recording(self):
        if (
            isinstance(self.recording_process, Process)
            and self.recording_process.is_alive()
        ):
            print("Please stop recording before starting again...")
        else:
            print("Recording started")
            self.recording_process = Process(target=record_and_detect)
            self.recording_process.start()

    def _stop_recording(self):
        try:
            self.recording_process.terminate()
            self.recording_process = None
            print("Eecording ended")
        except AttributeError:
            print("Please start the recording before ending it...")

    def _plot_audio(self):
        print()


def main():
    app = QApplication(sys.argv)
    window = RealTimeDetector()
    window.show()
    app.exec()
    sys.exit()


if __name__ == "__main__":
    main()
