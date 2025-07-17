from __future__ import print_function
import argparse
import os
import random
import sys
import wave

import librosa
import numpy as np
import pcen
import pyaudio

class AudioPreprocessor(object):
    def __init__(self, sr=16000, n_dct_filters=40, n_mels=40, f_max=4000, f_min=20, n_fft=480, hop_ms=10):
        super().__init__()
        self.n_mels = n_mels
        # self.dct_filters = librosa.filters.dct(n_dct_filters, n_mels)
        self.sr = sr
        self.f_max = f_max if f_max is not None else sr // 2
        self.f_min = f_min
        self.n_fft = n_fft
        self.hop_length = sr // 1000 * hop_ms
        self.pcen_transform = pcen.StreamingPCENTransform(n_mels=n_mels, n_fft=n_fft, hop_length=self.hop_length, trainable=True)
    
    def compute_mfccs(self, data):
        data = librosa.feature.melspectrogram(
            y=data,
            sr=self.sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            fmin=self.f_min,
            fmax=self.f_max
        )
        data[data >0 ] = np.log(data[data > 0])
        # data = [np.matmul(self.dct_filters, x) for x in np.split(data, data.shape[1], axis=1)]
        data = np.array(data, order="F").astype(np.float32)
        return data
    
    def compute_pcen(self, data):
        data = self.pcen_transform(data)
        self.pcen_transform.reset()
        return data

    