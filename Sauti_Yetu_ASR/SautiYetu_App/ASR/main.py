# !pip install gradio ffmpeg-python matplotlib numpy pandas scipy scikit-learn whisper librosa webrtcvad transformers datasets accelerate nltk sentence-transformers
# !pip install git+https://github.com/m-bain/whisperx.git
# !pip install torch==2.3.1 torchaudio==2.3.1 torchcodec==0.1.1 torchvision==0.18.1 --extra-index-url https://download.pytorch.org/whl/cu121


import string
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import torch
import whisper
# from dtw import dtw
from IPython.display import Audio, display
from scipy.ndimage import median_filter

