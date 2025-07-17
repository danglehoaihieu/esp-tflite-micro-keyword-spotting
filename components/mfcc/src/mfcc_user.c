#include "mfcc_user.h"
#include "kiss_fftr.h"

static const char *TAG = "MFCC";
// int16_t audio_buffer[MFCC_SAMPLE_RATE];                        // 1 second of audio
// int32_t log_mel_output[NUM_FRAMES][NUM_MEL_BINS];         // intermediate result
// float mfcc_output[NUM_FRAMES][NUM_MFCC_COEFFS];           // final MFCC result
static struct FrontendConfig config;
static struct FrontendState state;

void mfcc_init(void)
{
    FrontendFillConfigWithDefaults(&config);
    struct FilterbankConfig* fbank_config = &config.filterbank;
    fbank_config->lower_band_limit = MFCC_LOWER_EDGE_HERTZ;
    fbank_config->upper_band_limit = MFCC_UPPER_EDGE_HERTZ;

    config.window.size_ms = MFCC_FRAME_LENGTH_MS;         // 400
    config.window.step_size_ms = MFCC_FRAME_SHIFT_MS;        // 160
    config.filterbank.num_channels = MFCC_NUM_MEL_BINS;
    config.filterbank.lower_band_limit = MFCC_LOWER_EDGE_HERTZ;
    config.filterbank.upper_band_limit = MFCC_UPPER_EDGE_HERTZ;


    // config.noise_reduction.enable = false;      // match training
    config.pcan_gain_control.enable_pcan = 0;    // match training
    config.log_scale.enable_log = 1;             // log mel energies


    FrontendPopulateState(
        &config,
        &state,
        MFCC_SAMPLE_RATE
    );
    ESP_LOGI(TAG, "Init mfcc completed");
}
// Extract log-mel spectrogram features from 1-second audio buffer
void extract_log_mel_features(const int16_t* audio_data, int8_t out_features[MFCC_NUM_FRAMES][MFCC_NUM_MEL_BINS]) 
{
    // ESP_LOGI(TAG, "extract_log_mel_features start");
    // Loop over all overlapping frames in the audio buffer
    for (int i = 0; i < MFCC_NUM_FRAMES; i++) 
    {
        // Point to the start of the current frame using stride (FRAME_STEP)
        const int16_t* frame = &audio_data[i * MFCC_FRAME_STEP];
        // Will store how many samples were consumed (optional output)
        size_t samples_used = 0;
        // Call the Microfrontend to process the frame into log-mel features
        // `state` is assumed to be a global or previously initialized FrontendState
        // ESP_LOGI(TAG, "FrontendProcessSamples start");
        struct FrontendOutput output = FrontendProcessSamples(&state, frame, MFCC_FRAME_LEN, &samples_used);
        // ESP_LOGI(TAG, "WindowState state: size: %d, step: %d, input_used: %d, max_abs_output_value %d", state.window.size, state.window.step, state.window.input_used, state.window.max_abs_output_value);
        // ESP_LOGI(TAG, "FrontendState state: fft_size: %d, input_size: %d, scratch_size: %d", state.fft.fft_size, state.fft.input_size, state.fft.scratch_size);
        // ESP_LOGI(TAG, "filterbank state: num_channels: %d, start_index: %d, end_index: %d", state.filterbank.num_channels, state.filterbank.start_index, state.filterbank.end_index);


        // ESP_LOGI(TAG, "FrontendProcessSamples end");
        // If output is valid and contains the expected number of mel bins
        ESP_LOGI(TAG, "Extracted frame %d, valid=%d", i, output.values != NULL);
        if (output.values && output.size == MFCC_NUM_MEL_BINS) 
        {
            // ESP_LOGI(TAG, "output is valid and contains the expected number of mel bins");
            // Copy each log-mel bin value into the output array for this frame
            for (int j = 0; j < MFCC_NUM_MEL_BINS; j++) 
            {
                // Fixed-point Q-format int32_t
                // float f = output.values[j] * FRONTEND_SCALE;
                float f = MODEL_SCALE * (output.values[j] - ZERO_POINT);
                out_features[i][j] = (int8_t)round(f / MODEL_SCALE + ZERO_POINT);
            }
        } 
        else 
        {
            // ESP_LOGI(TAG, "output is invalid and zero out the entire row to avoid garbage data");
            // If output is invalid, zero out the entire row to avoid garbage data
            for (int j = 0; j < MFCC_NUM_MEL_BINS; j++) 
            {
                out_features[i][j] = ZERO_POINT;
            }
        }
    }
    // Clean up memory allocated by the Microfrontend (e.g. filterbank buffers)
    // ESP_LOGI(TAG, "Clean up memory allocated by the Microfrontend");
    // FrontendFreeStateContents(&state);
    // ESP_LOGI(TAG, "extract_log_mel_features done");
}