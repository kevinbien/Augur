# Augur 

Augur is a Python application for automatic zebra finch song detection. Using a lightweight CNN, it can quickly and reliably distinguish zebra finch song from other common noises encountered when recording zebra finches in the lab, no additional training required.

## Features

- Can automatically search folders of recordings for files containing song
- Can be used to detect and save zebra finch singing in real time
- Features a GUI for easy usage

## Installation

Augur can be easily installed using Conda. After cloning the repo, run 'cd Augur' to access the local repository. You can then run

```bash
conda create env create -n 'environment.yml'
```

to install the necessary packages and 

```bash
pip install .
```

to install Augur.

After installing the software, the GUI can be run from the terminal. Activate the Augur Conda environment using

```bash
conda activate augur
```

and open the GUI by running

```bash
augur
```

## Usage

When running Augur, the CNN takes 1 second windows from a recording and outputs the probability that each window contains song.

**Classification settings** are parameters the model uses when making predictions:

- Input channel: the channel read from when processing input recordings
- Threshold: the probability above which a window is considered to contain song
- Overlap: percentage by which windows used for prediction overlap
  - For example: when using 50% overlap, two 0.5s frames are combined into one prediction window




 
