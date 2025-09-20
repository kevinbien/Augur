from setuptools import setup, find_packages

setup(
    name="song_identifier",
    version="0.1",
    packages=find_packages(),
    install_requires=["pytorch", "librosa"],
    entry_points={
        "console_scripts": [
            "augur = augur_main.real_time_detector_pyqt:main",
        ],
    },
)
