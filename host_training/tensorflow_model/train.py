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
from model import create_model, create_model_v2
np.set_printoptions(precision=20, suppress=True) 
build = tf.sysconfig.get_build_info()
print("TensorFlow Build Info:")
for k, v in build.items():
    print(f"{k}: {v}")

print(tf.config.list_physical_devices('GPU'))

data_dir = "../data/speech_commands_v0.02"
testing_list_path = "D:/dev/keywordspotting/esp-tflite-micro_example/host_training/data/speech_commands_v0.02/testing_list.txt"
validation_list_path = "D:/dev/keywordspotting/esp-tflite-micro_example/host_training/data/speech_commands_v0.02/validation_list.txt"

testing_list_path = r"D:\dev\keywordspotting\esp-tflite-micro_example\host_training\data\speech_commands_v0.02\testing_list.txt"
validation_list_path = r"D:\dev\keywordspotting\esp-tflite-micro_example\host_training\data\speech_commands_v0.02\validation_list.txt"

# testing_list_path = "file://" + testing_list_path
# validation_list_path = "file://" + validation_list_path
perLabel_limit = int(11000/36 * 7) # 305 * n
test_perLabel_limit = int (perLabel_limit * 0.1)
val_perLabel_limit = int (perLabel_limit * 0.3)

train_perLabel_limit_count = {}
test_perLabel_limit_count = {}
val_perLabel_limit_count = {}

target_classes = ["cat", "go"] # auto include "unknown"
target_classes = target_classes + ["unknown"]
target_classes_index = {name: idx for idx, name in enumerate(target_classes)}

targetSample=16000
BATCH_SIZE = 16
EPOCHS = 50
PATIENCE = 5



#Load a WAV file
def load_audio_tensor(filePath, label):
    audio = tf.io.read_file(filePath)
    pcm, sample_rate = tf.audio.decode_wav(audio, desired_channels = 1)
    pcm  = tf.squeeze(pcm , axis = -1) # remove channel dim
    
    # Pad or trim
    pcm_len = tf.shape(pcm)[0]
    pcm = tf.cond(
        pcm_len < targetSample,
        lambda: tf.pad(pcm, [[0, targetSample - pcm_len]]),
        lambda: pcm[:targetSample]
    )
    
    log_mel_spectrogram = extract_mfcc(pcm, targetSample)
    log_mel_spectrogram = tf.expand_dims(log_mel_spectrogram, axis=-1)             # → (49, 13, 1)
    return log_mel_spectrogram, label

def extract_mfcc(audio_tensor, sample_rate, 
                  frame_len=640, frame_step=320, num_mel_bins=80, num_coeffs=13, fft_len=512,
                  lower_edge_hertz=80.0, upper_edge_hertz=7600.0):
    # Compute STFT
    stft = tf.signal.stft(audio_tensor, frame_length=frame_len, 
                          frame_step=frame_step, fft_length=fft_len,
                          window_fn=tf.signal.hann_window)
    spectrogram = tf.abs(stft)

    # Mel-scale filter bank
    mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mel_bins,
        num_spectrogram_bins=stft.shape[-1],
        sample_rate=sample_rate,
        lower_edge_hertz=lower_edge_hertz,
        upper_edge_hertz=upper_edge_hertz
    )
    mel_spectrogram = tf.matmul(spectrogram, mel_weight_matrix)

    # Log-scale
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.1920928955078125e-07)

    # MFCCs
    # mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :num_coeffs]
    return log_mel_spectrogram




#Extract MFCC using TFIO
def extract_mfcc_in_batch (audio_tensor_batch, sample_rate, 
                  frame_len=640, frame_step=320, num_mel_bins=40, num_coeffs=13, fft_len=640,
                  lower_edge_hertz=80.0, upper_edge_hertz=7600.0):
    

    # A Tensor of [batch_size, num_samples] mono PCM samples in the range [-1, 1].
    #pcm = tf.random.normal([batch_size, num_samples], dtype=tf.float32)

    # batch_size = tf.shape(audio_tensor_batch)[0] # batch size of pcm
    #compute spectrogram
    stfts = tf.signal.stft(audio_tensor_batch, frame_length=frame_len, frame_step=frame_step, fft_length=fft_len)
    spectrograms = tf.abs(stfts)

    #compute mel-scale filterbank
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix \
        (num_mel_bins, stfts.shape[-1], sample_rate, lower_edge_hertz, upper_edge_hertz)
    
    mel_spectrogram = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)

    mel_spectrogram.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    # Compute MFCCs from log_mel_spectrograms and take the first num_coeffs=13.
    # mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :num_coeffs]
    return log_mel_spectrogram

# # Full pipeline
# file_path = r"..\data\speech_commands_v0.02\backward\0a2b400e_nohash_0.wav"
# audio_tensor, sample_rate = load_audio_tensor(file_path)


# # Show audio
# # convert tensor into float number and show audio clip in graph
# audio_fl = tf.cast(audio_tensor, tf.float32) / 32768.0



# # Extract MFCCs
# mfccs = extract_mfcc(audio_tensor=audio_tensor, sample_rate=sample_rate)

# print("MFCC shape:", mfccs.shape)

# plt.figure()
# plt.plot(audio_fl.numpy())
# plt.title("Audio Waveform")
# plt.xlabel("Sample")
# plt.ylabel("Amplitude")
# plt.show()
def get_classes(data_dir):
    classes = np.array(tf.io.gfile.listdir(str(data_dir)))
    classes = classes[(classes != os.path.basename(validation_list_path)) & \
                    (classes != os.path.basename(testing_list_path)) & \
                    (classes != 'README.md') & (classes != '.DS_Store') & \
                    (classes != 'LICENSE')    ]

    return classes

def get_path_and_label(type_, fileListPath):
    global target_classes_index, train_perLabel_limit_count, test_perLabel_limit_count, val_perLabel_limit_count, target_classes
    paths = []
    labels = []
    with open(fileListPath, "r") as f:
        for line in f:
            path = os.path.dirname(fileListPath) + "/" +line.strip()
            # if not check_wav_files(path): continue
            class_name = os.path.basename(os.path.dirname(path))
            if (type_ == 'test') and test_perLabel_limit_count[class_name] > 0:
                test_perLabel_limit_count[class_name] -= 1
                paths.append(path)
                labels.append(target_classes_index.get(class_name, target_classes_index["unknown"]))

            elif (type_ == 'val') and val_perLabel_limit_count[class_name] > 0:
                val_perLabel_limit_count[class_name] -= 1
                paths.append(path)
                labels.append(target_classes_index.get(class_name, target_classes_index["unknown"]))
            
            if all(v == 0 for v in test_perLabel_limit_count.values()) or \
               all(v == 0 for v in val_perLabel_limit_count.values()): \
                return paths, labels
    
    return paths, labels


def findLabel(class_names, path):
    for label in class_names:
        if (("/"+label+"/") in path) or (("\\"+label+"\\") in path):
            return label
    return None

def existingDataset(existingPaths, label, name):
    path1 = label+"/"+name
    path2 = label+"\\"+name
    return any(path1 in item for item in existingPaths) or any(path2 in item for item in existingPaths)

def get_train_path_and_label(test_paths, val_paths, class_to_index): # get all files that excluded in test and val list
    global target_classes_index, train_perLabel_limit_count, test_perLabel_limit_count, val_perLabel_limit_count, target_classes
    all_files = []
    labels = []
    existingPaths = test_paths + val_paths

    for root, dirs, files in os.walk(data_dir):
        for name in files:
            full_path = os.path.join(root, name)
            label = findLabel(class_to_index.keys(), full_path)
            # print(full_path, label)
            if (label != None) and (train_perLabel_limit_count[label] > 0):
                if not check_wav_files(full_path): continue
                # check no belong to test and val list
                if not existingDataset(existingPaths, label, name):
                    print(f"{label} {train_perLabel_limit_count[label]:<10}", end='\r', flush=True)
                    all_files.append(full_path)
                    # labels.append(class_to_index[label])
                    labels.append(target_classes_index.get(label, target_classes_index["unknown"]))
                    train_perLabel_limit_count[label] -=1
            if all(v == 0 for v in train_perLabel_limit_count.values()): return all_files, labels

    return all_files, labels


def make_dataset(paths, labels, batch_size=32, shuffle=True):
    # paths = tf.constant(paths, dtype=tf.string)
    # labels = tf.constant(labels, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths))

    ds = ds.map(load_audio_tensor, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def check_wav_files(path):
    try:
        audio = tf.io.read_file(path)
        pcm, _ = tf.audio.decode_wav(audio, desired_channels=1)
    except Exception as e:
        print(f"❌ Error with file: {path}\n{e}")
        return False
    return True

def main():
    global train_perLabel_limit_count, test_perLabel_limit_count, val_perLabel_limit_count, target_classes
    print("main starts")
    
    all_class_names = get_classes(data_dir)
    # class_names = classes
    perLabel_limit_ = perLabel_limit // (len(all_class_names) - len(target_classes) + 1)
    test_perLabel_limit_ = test_perLabel_limit // (len(all_class_names) - len(target_classes) + 1)
    val_perLabel_limit_ = val_perLabel_limit // (len(all_class_names) - len(target_classes) + 1)

    class_to_index = {name: idx for idx, name in enumerate(all_class_names)}
    train_perLabel_limit_count = {name: perLabel_limit if name in target_classes else perLabel_limit_ for idx, name in enumerate(all_class_names)}
    test_perLabel_limit_count = {name: test_perLabel_limit if name in target_classes else test_perLabel_limit_ for idx, name in enumerate(all_class_names)}
    val_perLabel_limit_count = {name: val_perLabel_limit if name in target_classes else val_perLabel_limit_ for idx, name in enumerate(all_class_names)}

    print('train_perLabel_limit_count:', train_perLabel_limit_count)
    print('test_perLabel_limit_count:', test_perLabel_limit_count)
    print('val_perLabel_limit_count:', val_perLabel_limit_count)

    print("get paths, labels for val and test")
    test_paths, test_labels = get_path_and_label(type_='test', fileListPath=testing_list_path)
    val_paths, val_labels = get_path_and_label(type_='val', fileListPath=validation_list_path)

    print(test_paths[:5])
    print(test_labels[:5])
    print("----------------")
    print(val_paths[:5])
    print(val_labels[:5])

    # get paths and labels for train
    print("get paths and labels for train")
    train_paths, train_labels = get_train_path_and_label(test_paths, val_paths, class_to_index)

    print("----------------")
    print(train_paths[:5])
    print(train_labels[:5])

    # make dataset
    print("make dataset for train")
    train_dataset = make_dataset(train_paths, train_labels)
    print("make dataset for val")
    val_dataset = make_dataset(val_paths, val_labels, batch_size=BATCH_SIZE, shuffle=False)
    print("make dataset for test")
    test_dataset = make_dataset(test_paths, test_labels, batch_size=BATCH_SIZE, shuffle=False)

    print("train: ", len(train_paths))
    print("test: ", len(test_paths))
    print("val: ", len(val_paths))

    input_shape = None
    for x, y in train_dataset.take(1):
        input_shape = x.shape[1:]  # Exclude batch dimension
        print("Input shape:", input_shape)


    model = create_model_v2(input_shape, len(target_classes))
    model.summary()

    early_stop = EarlyStopping(
        monitor='val_loss',       # or 'val_accuracy'
        patience=PATIENCE,               # wait 5 epochs with no improvement
        restore_best_weights=True # go back to best weights after stop
    )

    checkpoint = ModelCheckpoint(
        filepath='best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,     # Set True if you only want weights
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[early_stop, checkpoint]
    )


    # # 3. Quantize the model using TFMOT
    # quantize_model = tfmot.quantization.keras.quantize_model
    # q_aware_model = quantize_model(model)

    # q_aware_model.summary()

    # # 4. Re-compile the quantized model (must do this!)
    # q_aware_model.compile(
    #     optimizer=tf.keras.optimizers.Adam(),
    #     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #     metrics=['accuracy'],
    # )


    # # Optionally re-train for better accuracy
    # history = q_aware_model.fit(
    #     train_dataset,
    #     validation_data=val_dataset,
    #     epochs=EPOCHS,
    #     callbacks=[early_stop, checkpoint]
    # )

    # # Save quantized model
    # q_aware_model.save("./best_quantized_model")

    metrics = history.history
    plt.figure(figsize=(16,6))
    plt.subplot(1,2,1)
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel('Epoch')
    plt.ylabel('Loss [CrossEntropy]')

    plt.subplot(1,2,2)
    plt.plot(history.epoch, 100*np.array(metrics['accuracy']), 100*np.array(metrics['val_accuracy']))
    plt.legend(['accuracy', 'val_accuracy'])
    plt.ylim([0, 100])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [%]')
    # plt.show(block=False)


    model.evaluate(test_dataset, return_dict=True)
    y_pred = model.predict(test_dataset)
    y_pred = tf.argmax(y_pred, axis=1)
    y_true = tf.concat(list(test_dataset.map(lambda s,lab: lab)), axis=0)
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx,
                xticklabels=target_classes,
                yticklabels=target_classes,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()

    print("convert to tflite model")
    convert(model, train_dataset)

    
def get_sample():
    file_path = r"..\data\speech_commands_v0.02\backward\0a2b400e_nohash_0.wav"


    audio = tf.io.read_file(file_path)
    pcm, sample_rate = tf.audio.decode_wav(audio, desired_channels=1)
    pcm = tf.squeeze(pcm, axis=-1)

    # Chuyển thành int16: float32 [-1.0, 1.0) -> int16 [-32768, 32767]
    pcm_int16 = np.clip(pcm.numpy() * 32768, -32768, 32767).astype(np.int16)
    # Ghi ra file C header format
    with open(r"D:\dev\keywordspotting\esp-tflite-micro_example\main\include\inference_buffer.h", "w") as f:
        f.write(f"static int16_t inference_buffer[{len(pcm_int16)}] = {{\n")
        for i, val in enumerate(pcm_int16):
            f.write(f"{val}, ")
            if (i + 1) % 16 == 0:  # Xuống dòng mỗi 16 giá trị cho dễ đọc
                f.write("\n")
        f.write("\n};\n")
    # log_mel_spectrogram, _ = load_audio_tensor(filePath=file_path, label="backward")
    # print(log_mel_spectrogram)


def get_feature_sample():
    file_path = r"..\data\speech_commands_v0.02\backward\0a2b400e_nohash_0.wav"
    audio = tf.io.read_file(file_path)
    pcm, sample_rate = tf.audio.decode_wav(audio, desired_channels=1)
    pcm = tf.squeeze(pcm, axis=-1)
    log_mel_spectrogram = extract_mfcc(pcm, sample_rate)
    log_mel_spectrogram = tf.expand_dims(log_mel_spectrogram, axis=-1)

    scale = 0.07797757536172867
    zero_point = 49
    mfcc_int8 = np.round(log_mel_spectrogram / scale + zero_point).astype(np.int32)
    mfcc_int8 = np.clip(mfcc_int8, -128, 127).astype(np.int8)

    print(mfcc_int8)


def convert(model, dataset):
    # # Required to recognize Quantize wrappers and configs
    # custom_objects = tfmot.quantization.keras.quantize_scope().__enter__()

    # # Load the quantized model
    # with tfmot.quantization.keras.quantize_scope():
    #     model = tf.keras.models.load_model("best_model.h5")

    def representative_dataset():
        print("quantization calibration using dataset...")
        for batch, _ in dataset.take(100):  # Replace with your dataset
            for input_value in batch:
                # print(input_value.shape, input_value.dtype)
                yield [tf.expand_dims(tf.cast(input_value, tf.float32), axis=0)]

    # print("input_shape of model:", model.input_shape)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter = tf.lite.TFLiteConverter.from_saved_model("./best_quantized_model")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_type = tf.int8
    
    tflite_model = converter.convert()
    print("✅ Model conversion completed!")
    # Save for ESP32
    with open(r"D:\dev\keywordspotting\esp-tflite-micro_example\components\tflite_model\model.tflite", "wb") as f:
        f.write(tflite_model)

if __name__ == "__main__":
    main()
    # get_sample()
    # get_feature_sample()
    # path = r"..\data\speech_commands_v0.02\backward\0a2b400e_nohash_0.wav"
    # f, _ = load_audio_tensor(path, 1)
    # print(f.numpy()[0])