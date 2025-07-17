import os
import tensorflow as tf
import tensorflow_io as tfio  # for tf.audio
import numpy as np
import librosa
import glob

import matplotlib.pyplot as plt
import seaborn as sns

def check_wav_files(file_paths):
    for path in file_paths:
        try:
            audio = tf.io.read_file(path)
            pcm, _ = tf.audio.decode_wav(audio, desired_channels=1)
        except Exception as e:
            print(f"❌ Error with file: {path}\n{e}")


check_wav_files()