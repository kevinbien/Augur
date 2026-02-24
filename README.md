# Augur 

Augur is a Python application for automatic zebra finch song detection. Using a lightweight CNN, it can distinguish zebra finch song from other common noises encountered when recording zebra finches in the lab, without requiring additional training by the user.

*The underlying CNN was trained on recordings from 7 adult zebra finches. Validation data will be presented in a future update.* 

## Features

- Can automatically search folders of recordings for files containing song
- Can be used to detect and save zebra finch singing in real time
- Includes a GUI for easy use

## Installation

Augur can be easily installed on Windows using Conda. After cloning the repo, run 'cd Augur' to access the local repository. You can then run

```bash
conda env create -f environment.yml
```

to install the necessary packages and 

```bash
pip install .
```

to install Augur.

After installing the software, the GUI can be run from the terminal. Activate the Conda environment using

```bash
conda activate augur
```

and open the GUI by running

```bash
augur
```

## Usage

Internally, the model takes 1 second windows from a recording and outputs the probability that each window contains song. 

**Classification settings** are parameters the model uses when making predictions:

- Input channel: the channel read from when processing input recordings
  - The input channel is zero indexed; mono files always use channel 0 
- Threshold: the probability above which a window is considered to contain song
 
**Input/output locations** are directories that Augur reads from and writes to. Augur processes .wav files from the input directory and all its subdirectories, and copies any file determined to contain song to a local "Found Song" subdirectory, and to the output directory if provided.

**Live detection** can be performed by choosing an input device from the drop down menu and pressing "Start recording". After opening the recording, the model will read 0.5s chunks from the input device's stream and output predictions in the terminal. During live recording, any bout of song detected is saved to the output directory along with the 5s of audio preceding and following the bout. 

*Live detection is still in development and will be updated for performance and usability*

## Why "Augur"?

[Augury](https://en.wikipedia.org/wiki/Augury) was the ancient practice of divining the future by reading omens from birds. This software can't do that, but machine learning can sometimes feel like magic.






 
