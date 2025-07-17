#include <stdio.h>
#include <stdlib.h>
#include "string.h"
#include <limits.h>
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_mfcc_iface.h"
#include "esp_mfcc_models.h"

#include "tflite_model_shared.h"

#define MFCC_NFFT               512
#define MFCC_SAMPLE_RATE        16000        // 16 kHz audio
#define MFCC_FRAME_LENGTH_MS    40           // each frame: 40 ms
#define MFCC_FRAME_SHIFT_MS     20           // stride: 20 ms
#define MFCC_NUM_MEL_BINS       80           // output from frontend
#define MFCC_NUM_MFCC_COEFFS    13           // DCT coeffs to keep
#define MFCC_AUDIO_DURATION_MS  1000         // process 1 second
#define MFCC_FRAME_LEN          ((MFCC_SAMPLE_RATE * MFCC_FRAME_LENGTH_MS) / 1000)  // 640
#define MFCC_FRAME_STEP         ((MFCC_SAMPLE_RATE * MFCC_FRAME_SHIFT_MS) / 1000)   // 320
#define MFCC_NUM_FRAMES         (((MFCC_SAMPLE_RATE * MFCC_AUDIO_DURATION_MS / 1000 - MFCC_FRAME_LEN) / MFCC_FRAME_STEP) + 1) // 49
#define MFCC_LOWER_EDGE_HERTZ    80.0
#define MFCC_UPPER_EDGE_HERTZ    7600.0
#define MFCC_PREEMPHASIS_FILTER_COEFF   0 // no apply

// #define TFLITE_MODE_SCALE_FACTOR    0.08371087908744812
// #define TFLITE_MODE_ZERO_POINT      62

#ifdef __cplusplus
extern "C" {
#endif
int compute_mfcc_features(const int16_t* audio_frame, float* out_features);
int compute_quantized_int8_mfcc_features(const int16_t* audio_frame, int8_t* out_features);
// Initializes the MFCC frontend. Must be called before compute.
int mfcc_frontend_init(void);
// Frees resources used by MFCC frontend
void mfcc_frontend_destroy(void);
#ifdef __cplusplus
}
#endif