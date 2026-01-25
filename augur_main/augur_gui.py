import sys
from multiprocessing import Process, shared_memory
import sounddevice as sd
from screeninfo import get_monitors
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QSizePolicy,
    QWidget,
    QGridLayout,
    QPushButton,
    QComboBox,
    QFileDialog,
    QLabel,
    QLineEdit,
)
import pyqtgraph as pg
import librosa

from augur_main.functions import process_folder, record_and_detect


class AugurGUI(QWidget):
    def __init__(self):
        super().__init__()

        # Set up the window
        self.setWindowTitle("Augur 0.5")
        width, height = get_monitors()[0].width // 4, get_monitors()[0].height // 3
        self.setGeometry(0, 0, width, height)

        # Set the layout for the window
        layout = QGridLayout()
        self.setLayout(layout)

        # Set processes to none until user starts process
        self.recording_process = None
        self.detecting_process = None
        self.shm = None
        self.recording_window = None

        # Set folder paths to None until user chooses
        self.song_loc = None
        self.song_dest = None

        # Create buttons
        self.ofolder_button = QPushButton("Choose output folder", self)
        self.ifolder_button = QPushButton("Choose input folder", self)
        self.start_button = QPushButton("Start recording", self)
        self.stop_button = QPushButton("Stop recording", self)
        self.detect_button = QPushButton("Detect song from existing folder")
        self.start_button.clicked.connect(self._start_recording)
        self.stop_button.clicked.connect(self._stop_recording)
        self.detect_button.clicked.connect(self._detect_song)
        self.ifolder_button.clicked.connect(self._choose_folder)
        self.ofolder_button.clicked.connect(self._choose_folder)
        layout.addWidget(self.ifolder_button, 5, 1)
        layout.addWidget(self.ofolder_button, 6, 1)
        layout.addWidget(self.detect_button, 7, 0, 1, 2)
        font = self.detect_button.font()
        font.setBold(True)
        self.detect_button.setFont(font)
        layout.addWidget(self.start_button, 10, 0)
        layout.addWidget(self.stop_button, 10, 1)

        # Create labels
        self.settings_label = QLabel("Classification settings", self)
        self.channel_label = QLabel("Input channel:", self)
        self.threshold_label = QLabel("Threshold:", self)
        self.overlap_label = QLabel("Overlap:", self)
        self.folders_label = QLabel("Input/output locations", self)
        self.recording_label = QLabel("Detect song during live recording", self)
        self.ifolder_label = QLabel("No input folder selected:", self)
        self.ofolder_label = QLabel("No output folder selected:", self)
        self.device_label = QLabel("Select input device:", self)
        layout.addWidget(self.settings_label, 0, 0, 1, 2)
        layout.addWidget(self.channel_label, 1, 0)
        layout.addWidget(self.threshold_label, 2, 0)
        layout.addWidget(self.overlap_label, 3, 0)
        layout.addWidget(self.folders_label, 4, 0, 1, 2)
        layout.addWidget(self.ifolder_label, 5, 0)
        layout.addWidget(self.ofolder_label, 6, 0)
        layout.addWidget(self.recording_label, 8, 0, 1, 2)
        layout.addWidget(self.device_label, 9, 0)
        font = self.folders_label.font()
        font.setBold(True)
        self.settings_label.setFont(font)
        self.folders_label.setFont(font)
        self.recording_label.setFont(font)
        self.settings_label.setAlignment(Qt.AlignCenter)
        self.folders_label.setAlignment(Qt.AlignCenter)
        self.recording_label.setAlignment(Qt.AlignCenter)

        # Create input comboboxes
        self.device_box = QComboBox()
        for device in sd.query_devices():
            if device["max_input_channels"] > 0:
                self.device_box.addItem(device["name"], device["index"])
        if self.device_box.count() == 0:
            self.device_box.setCurrentText("No input device found...")
            self.start_button.setDisabled(True)
        self.overlap_box = QComboBox()
        self.overlap_box.addItems([r"0% overlap", r"50% overlap", r"75% overlap"])
        self.overlap_box.setCurrentText(r"50% overlap")
        layout.addWidget(self.device_box, 9, 1)
        layout.addWidget(self.overlap_box, 3, 1)

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
            self.detect_button,
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
            model_path = Path(__file__).resolve().parent / "model_1.0_0.0346.pt"

            # Create shared memory
            rate = 22050
            max_seconds = 60
            array_size = rate * max_seconds
            self.shm = shared_memory.SharedMemory(
                create=True, size=np.zeros((2, array_size), dtype=np.float32).nbytes
            )

            self.recording_window = RecordingWindow(self.shm.name, rate, max_seconds)
            self.recording_window.show()

            self.recording_process = Process(
                target=record_and_detect,
                args=(
                    model_path,
                    input_device,
                    self.song_dest,
                    self.shm.name,
                    rate,
                    max_seconds,
                ),
            )
            self.recording_process.start()

    def _detect_song(self):
        if (
            isinstance(self.detecting_process, Process)
            and self.detecting_process.is_alive()
        ):
            print(
                "Please wait for detecting process to finish before starting again..."
            )
        if self.song_loc is None:
            print("Please provide an input folder before detecting for song")
        else:
            try:
                model_path = Path(__file__).resolve().parent / "model_1.0_0.0346.pt"

                if self.overlap_box.currentText() == r"0% overlap":
                    overlap_windows = 1
                elif self.overlap_box.currentText() == r"50% overlap":
                    overlap_windows = 2
                else:
                    overlap_windows = 4

                self.detecting_process = Process(
                    target=process_folder,
                    args=(
                        model_path,
                        self.song_loc,
                        self.song_dest,
                        int(self.channel_text.text()),
                        float(self.threshold_text.text()),
                        overlap_windows,
                    ),
                )
                self.detecting_process.start()
            except:
                print(
                    "Please make sure you have correctly specified the input folder..."
                )

    def _stop_recording(self):
        try:
            self.recording_process.terminate()
            self.recording_process = None

            self.recording_window.close()
            self.recording_window = None

            self.shm.close()
            self.shm.unlink()
            self.shm = None

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
    def __init__(self, shm_name, rate, max_seconds):
        super().__init__()

        # Set up the window
        self.setWindowTitle(" GUI")
        width, height = get_monitors()[0].width // 2, get_monitors()[0].height // 3
        layout = QGridLayout(self)
        self.setGeometry(0, 0, width, height)

        # Add toggle buttons
        self.toggle_raw = QPushButton("Waveform view")
        self.toggle_raw.clicked.connect(self._toggle_raw)
        layout.addWidget(self.toggle_raw, 0, 0)
        self.toggle_rms = QPushButton("RMS view")
        self.toggle_rms.clicked.connect(self._toggle_rms)
        layout.addWidget(self.toggle_rms, 0, 1)
        self.toggle_spec = QPushButton("Prediction view")
        self.toggle_spec.clicked.connect(self._toggle_pred)
        layout.addWidget(self.toggle_spec, 0, 2)
        self.toggle_spec = QPushButton("Spectrogram view")
        self.toggle_spec.clicked.connect(self._toggle_spec)
        layout.addWidget(self.toggle_spec, 0, 3)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.setTitle("Live Data Stream", color="k", size="12pt")
        self.plot_widget.setLabel("left", "Value", color="k")
        self.plot_widget.setLabel("bottom", "Sample", color="k")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.plot_widget, 1, 0, 1, 4)

        # Load audio from recording
        self.rate = rate
        self.shm = shared_memory.SharedMemory(name=shm_name)
        self.audio = np.ndarray(
            (2, self.rate * max_seconds), dtype=np.float32, buffer=self.shm.buf
        )
        self.plotted_audio = np.zeros(self.rate)

        self.mode = "raw"

        # Create waveform plot
        pen = pg.mkPen(color="b", width=2)
        self.wave_plot = self.plot_widget.plot(self.plotted_audio, pen=pen)

        # Create spectrogram image plot
        self.spec_plot = pg.ImageItem()
        self.plot_widget.addItem(self.spec_plot)
        self.spec_plot.setVisible(False)

        # Set up timer for updates
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_plot)
        self.timer.start(500)  # Update every 50ms

    def _update_plot(self):

        if self.mode == "raw" or self.mode == "rms":
            reshaped = self.audio[0].reshape(len(self.audio[0]) // 1000, 1000)
            if self.mode == "raw":
                self.plotted_audio = reshaped.mean(axis=1)
            elif self.mode == "rms":
                self.plotted_audio = np.sqrt((reshaped**2).mean(axis=1))
            self.wave_plot.setData(self.plotted_audio)
        elif self.mode == "pred":
            reshaped = self.audio[1].reshape(len(self.audio[0]) // 1000, 1000)
            self.plotted_audio = reshaped.mean(axis=1)
            self.wave_plot.setData(self.plotted_audio)
        else:
            mels = librosa.feature.melspectrogram(
                y=self.audio[0],
                n_fft=1024,
                hop_length=int(self.rate / 128),
                fmin=300,
                fmax=10000,
            )
            mels = librosa.power_to_db(mels)
            self.spec_plot.setImage(mels.T, autoLevels=True)

    def _toggle_raw(self):
        self.mode = "raw"
        self.plot_widget.enableAutoRange()
        self.spec_plot.setVisible(False)
        self.wave_plot.setVisible(True)

    def _toggle_rms(self):
        self.mode = "rms"
        self.plot_widget.enableAutoRange()
        self.spec_plot.setVisible(False)
        self.wave_plot.setVisible(True)

    def _toggle_pred(self):
        self.mode = "pred"
        self.plot_widget.setYRange(0, 1, padding=0)
        self.spec_plot.setVisible(False)
        self.wave_plot.setVisible(True)

    def _toggle_spec(self):
        self.mode = "spec"
        self.plot_widget.enableAutoRange()
        self.spec_plot.setVisible(True)
        self.wave_plot.setVisible(False)

    def closeEvent(self, event):
        self.timer.stop()
        self.shm.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = AugurGUI()
    window.show()
    app.exec()
    sys.exit()


if __name__ == "__main__":
    main()
