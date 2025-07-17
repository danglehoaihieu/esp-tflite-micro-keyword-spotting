import os
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import load_model

import numpy as np

# model = tf.keras.models.load_model("best_model.h5")

# model = load_model(
#     "qat_model.h5",
#     custom_objects={
#         "QuantizeLayer": tfmot.quantization.keras.QuantizeLayer,
#         # Add other custom layers/configs here if used
#     }
# )

# Required to recognize Quantize wrappers and configs
custom_objects = tfmot.quantization.keras.quantize_scope().__enter__()

# Load the quantized model
with tfmot.quantization.keras.quantize_scope():
    model = tf.keras.models.load_model("best_model.h5")



# Hàm tạo dữ liệu đại diện để calibrate (phải giống input thật)
def representative_data_gen():
    for i in range(100):
        # Giả sử input shape là (49, 40), bạn thay bằng shape thực tế
        data = np.random.rand(1, 49, 40, 1).astype(np.float32)
        yield [data]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

# Save for ESP32
with open(r"D:\dev\keywordspotting\esp-tflite-micro_example\components\tflite_model\model.tflite", "wb") as f:
    f.write(tflite_model)