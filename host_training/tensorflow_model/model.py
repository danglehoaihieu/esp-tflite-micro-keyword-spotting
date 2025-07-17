import os
import tensorflow as tf
import tensorflow_io as tfio  # for tf.audio
import numpy as np
import librosa
import glob

import matplotlib.pyplot as plt

def create_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Reshape((-1, 1)),  # Expand for Conv1D if input is 1D
        tf.keras.layers.Conv1D(16, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(),
        tf.keras.layers.Conv1D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def create_model_v2(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        # Downsample the input.
        # tf.keras.layers.Resizing(32, 32),
        # tf.keras.layers.Reshape((*input_shape, 1)),           # → (time_steps, mfcc, 1)
        # # Normalize.
        # norm_layer,
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Flatten(),
        
        tf.keras.layers.Dense(128, activation='relu'),

        tf.keras.layers.Dense(num_classes),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    return model
    
