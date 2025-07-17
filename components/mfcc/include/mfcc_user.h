#include "frontend.h"
#include "frontend_util.h"

#include "esp_log.h"

#define MFCC_SAMPLE_RATE        16000        // 16 kHz audio
#define MFCC_FRAME_LENGTH_MS    40           // each frame: 40 ms
#define MFCC_FRAME_SHIFT_MS     20           // stride: 20 ms
#define MFCC_NUM_MEL_BINS       40           // output from frontend
#define MFCC_NUM_MFCC_COEFFS    13           // DCT coeffs to keep
#define MFCC_AUDIO_DURATION_MS  1000         // process 1 second
#define MFCC_FRAME_LEN          ((MFCC_SAMPLE_RATE * MFCC_FRAME_LENGTH_MS) / 1000)  // 640
#define MFCC_FRAME_STEP         ((MFCC_SAMPLE_RATE * MFCC_FRAME_SHIFT_MS) / 1000)   // 320
#define MFCC_NUM_FRAMES         (((MFCC_SAMPLE_RATE * MFCC_AUDIO_DURATION_MS / 1000 - MFCC_FRAME_LEN) / MFCC_FRAME_STEP) + 1) // 49
#define MFCC_LOWER_EDGE_HERTZ    80.0
#define MFCC_UPPER_EDGE_HERTZ    7600.0


#define FRONTEND_SCALE (1.0f / (1 << FIXED_POINT))  // = 1/256
#define MODEL_SCALE   0.07797757f
#define ZERO_POINT    49




#ifdef __cplusplus
extern "C" {
#endif
// void initialize_mfcc_extract(void);
// int extract_mfcc(int16_t *pcm_input, float mfcc_output[NUM_FRAMES][NUM_MFCC_COEFFS]);
void mfcc_init(void);
void extract_log_mel_features(const int16_t* audio_data, int8_t out_features[MFCC_NUM_FRAMES][MFCC_NUM_MEL_BINS]);
#ifdef __cplusplus
}
#endif