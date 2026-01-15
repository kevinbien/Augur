# Augur 

Augur is a Python application for automatic zebra finch song detection. Using a lightweight CNN, it can quickly and reliably distinguish zebra finch song from other common noises encountered when recording zebra finches in the lab, no pretraining required.

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




