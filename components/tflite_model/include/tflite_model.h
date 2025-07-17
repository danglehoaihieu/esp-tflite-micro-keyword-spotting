#ifndef TFLITE_MODEL_H_
#define TFLITE_MODEL_H_

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "esp_log.h"
#include "esp_heap_caps.h"

#include "mfcc_sr.h"


#define TFLITE_MODEL_ACCEPT_THRESHOLD   0.75f
// The following values are derived from values used during model training.
// If you change the way you preprocess the input, update all these constants.
constexpr int kFeatureSize = MFCC_NUM_MEL_BINS; // map to mfcc
constexpr int kFeatureCount = MFCC_NUM_FRAMES;
constexpr int kFeatureElementCount = (kFeatureSize * kFeatureCount);
constexpr int kTensorArenaSize = 316 * 1024;
// Variables for the model's output categories.
constexpr int kCategoryCount = 3;
constexpr const char* kCategoryLabels[kCategoryCount] = {
    "cat",
    "go",
    "unknown",
};


void initialize_model(void);
int model_inference(int8_t* feature_buffer);

#endif