import os
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_io as tfio  # for tf.audio
import numpy as np
import librosa
import glob

import matplotlib.pyplot as plt
import seaborn as sns

from train import extract_mfcc
# Helper function to run inference on a TFLite model
def run_tflite_model(tflite_file):

    # Initialize the interpreter
    interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    file_path = r"..\data\speech_commands_v0.02\backward\0a2b400e_nohash_0.wav"
    audio = tf.io.read_file(file_path)
    pcm, sample_rate = tf.audio.decode_wav(audio, desired_channels=1)
    pcm = tf.squeeze(pcm, axis=-1)
    log_mel_spectrogram = extract_mfcc(pcm, sample_rate)
    log_mel_spectrogram = tf.expand_dims(log_mel_spectrogram, axis=-1)

    # Check if the input type is quantized, then rescale input data to int8
    if input_details['dtype'] == np.int8:
        input_scale, input_zero_point = input_details["quantization"]
        log_mel_spectrogram = log_mel_spectrogram / input_scale + input_zero_point
    # print(log_mel_spectrogram)
    log_mel_spectrogram = np.expand_dims(log_mel_spectrogram, axis=0).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], log_mel_spectrogram)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]


    print(output)
    # print(output_details['dtype'])
    # if output_details['dtype'] == np.int8:
    #     output_scale, output_zero_point = output_details["quantization"]
    #     float_output = output_scale * (output - output_zero_point)
    #     print(float_output)
    # else:
    #     print(output)

    ret = output.argmax()

    return ret

run_tflite_model(r"D:\dev\keywordspotting\esp-tflite-micro_example\components\tflite_model\model.tflite")
