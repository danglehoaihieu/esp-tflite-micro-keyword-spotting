# 🧠 esp-tflite-micro-keyword-spotting

A complete pipeline for training a TensorFlow model to perform **keyword spotting**, and deploying it on **ESP32 using TensorFlow Lite Micro** (TFLM). This project supports training, quantization, and integration with ESP-IDF.

---

## 📦 Features

- Train a lightweight keyword spotting (KWS) model with TensorFlow/Keras
- Export and quantize to TensorFlow Lite format
- Convert to C array for embedded inference
- Deploy and run on ESP32/ESP32-S3 using TFLM
- Real-time microphone inference (optionally with MFCC or log-mel frontend)

---

## 🗂️ Project Structure

```
esp-tflite-micro-keyword-spotting/
├── hos_training/tensorflow_model          # TensorFlow training scripts and notebooks
├── components/tflite_model            # Exported .tflite and C converted model
├── main/                    # ESP-IDF application source (main.cpp)
├── components/              # Optional: MFCC/log-mel frontend component
├── CMakeLists.txt           # ESP-IDF build system files
├── sdkconfig                # ESP-IDF config
└── README.md

---

## 🧪 1. Train the Model (on PC)

Install dependencies:

cuda:
TensorFlow Build Info:
cuda_compute_capabilities: ['sm_35', 'sm_50', 'sm_60', 'sm_70', 'sm_75', 'compute_80']
cuda_version: 64_112
cudart_dll_name: cudart64_112.dll
cudnn_dll_name: cudnn64_8.dll
cudnn_version: 64_8
is_cuda_build: True
is_rocm_build: False
is_tensorrt_build: False
msvcp_dll_names: msvcp140.dll,msvcp140_1.dll
nvcuda_dll_name: nvcuda.dll


Train, export, Quantize and convert to TensorFlow Lite format a model:
```
```bash
python train.py
```

---

## 🚀 3. Build and Flash to ESP32

Make sure you're using **ESP-IDF v5.3 or newer**.

```bash
idf.py set-target esp32s3
idf.py menuconfig     # (Optional: Enable PSRAM)
idf.py build
idf.py flash monitor
```

---

## 🎙️ 4. Real-Time Audio Inference

Supports:

- Built-in I2S microphone
- Real-time MFCC or log-mel extraction (ESP-SR frontend, custom MFCC)
- Keyword detection with probability thresholding

---

## 📚 References

- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)
- [ESP-SR (Speech Recognition)](https://github.com/espressif/esp-sr)
- [TensorFlow Audio Datasets](https://www.tensorflow.org/datasets/catalog/speech_commands)

---

## 🧑‍💻 Author

**Hieu Dang**  
GitHub: [@danglehoaihieu](https://github.com/danglehoaihieu)

---

## 🪪 License

MIT License. See [LICENSE](LICENSE) file for details.
