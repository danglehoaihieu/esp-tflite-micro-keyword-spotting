#include "mfcc_sr.h"

// Static model instance
static esp_mfcc_data_t* fbank_model = NULL;
// Backend interface
static const esp_mfcc_iface_t* fbank_handle = &esp_fbank_f32;

static const char *TAG = "MFCC_SR";

esp_mfcc_opts_t *get_fbank_opts_kaldi()
{
    // esp_mfcc_opts_t *opts = (esp_mfcc_opts_t*)malloc(sizeof(esp_mfcc_opts_t));
    static esp_mfcc_opts_t opts;  // <- static, no malloc!
    opts.psram_first = true;
    opts.use_power = false; // tf.abs(stft)
    opts.use_log_fbank = 1; 
    opts.log_epsilon = 1.1920928955078125e-07f; // torch.finfo(torch.float32).eps
    opts.win_type = (char *)("hanning"); // remove [-Wwrite-strings] warning
    opts.low_freq = MFCC_LOWER_EDGE_HERTZ;
    opts.high_freq = MFCC_UPPER_EDGE_HERTZ;
    opts.samp_freq = MFCC_SAMPLE_RATE;
    opts.nch = 1;
    opts.nfft = MFCC_NFFT;
    opts.nfilter = MFCC_NUM_MEL_BINS;
    opts.numcep = MFCC_NUM_MEL_BINS;
    opts.preemph = MFCC_PREEMPHASIS_FILTER_COEFF;//0.97;
    opts.append_energy = true;
    opts.winlen_ms = MFCC_FRAME_LENGTH_MS;
    opts.winstep_ms = MFCC_FRAME_SHIFT_MS;
    opts.remove_dc_offset = true;

    return &opts;
}

// Initializes the MFCC frontend. Must be called before compute.
int mfcc_frontend_init(void)
{
    if (fbank_model != NULL) 
    {
        ESP_LOGW(TAG, "MFCC frontend already initialized");
        return 0; // Already initialized
    }
    
    fbank_model = fbank_handle->create(get_fbank_opts_kaldi());
    if (!fbank_model) 
    {
        ESP_LOGE(TAG, "Failed to create MFCC model");
        return -1;
    }

    ESP_LOGI(TAG, "MFCC frontend initialized");
    return 0;
}

// Frees resources used by MFCC frontend
void mfcc_frontend_destroy(void)
{
    if (fbank_model) 
    {
        fbank_handle->destroy(fbank_model);
        fbank_model = NULL;
        ESP_LOGI(TAG, "MFCC frontend destroyed");
    }
}

/**
 * @brief Compute MFCC (log-mel filterbank) features for a single audio frame.
 *
 * @param audio_frame Pointer to full int16_t audio frame (16 kHz, 32ms).
 * @param out_features Pointer to fbank
 * @return void
 */
int compute_mfcc_features(const int16_t* audio_frame, float* out_features)
{
    // if (!fbank_model) {
    //     ESP_LOGI(TAG, "Failed to create fbank model");
    //     return -1;
    // }

    // Run feature extraction
    for (int i = 0; i< MFCC_NUM_FRAMES; i++)
    {
        int16_t* audio_feed = &audio_frame[i * MFCC_FRAME_STEP];
        float* temp_out = fbank_handle->run_step(fbank_model, audio_feed, 0);
        memcpy(&out_features[i * MFCC_NUM_MEL_BINS], temp_out, MFCC_NUM_MEL_BINS * sizeof(float));
    }

    // Destroy the model to release internal memory
    // fbank_handle->destroy(fbank_model);
    return 1;
}
/**
*@brief Compute MFCC (log-mel filterbank) features 
        and quantized base on scale and zero point of tflite model for a single audio frame.
*@param audio_frame pointer to full int16_t audio frame
*@param out_features Pointer to fbank (quantized int8)
*/
int compute_quantized_int8_mfcc_features(const int16_t* audio_frame, int8_t* out_features)
{
    // ESP_LOGI(TAG, "%f, %d", tflite_model_input_scale, tflite_model_input_zero_point);
    // Run feature extraction
    // float temp_buf[MFCC_NUM_MEL_BINS];
    for (int i = 0; i< MFCC_NUM_FRAMES; i++)
    {
        // ESP_LOGI(TAG, "1");
        // assert(heap_caps_check_integrity_all(true));  // check before
        int16_t* audio_feed = &audio_frame[i * MFCC_FRAME_STEP];

        // ESP_LOGI(TAG, "2");
        // assert(heap_caps_check_integrity_all(true));
        float* temp_out = fbank_handle->run_step(fbank_model, audio_feed, 0);
        // memcpy(out_features[i], temp_out, MFCC_NUM_MEL_BINS * sizeof(float));
        // memcpy(temp_buf, temp_out, MFCC_NUM_MEL_BINS * sizeof(float));
        // ESP_LOGI(TAG, "3");
        // assert(heap_caps_check_integrity_all(true));
        for (int j=0; j<MFCC_NUM_MEL_BINS; j++)
        {
            out_features[(i * MFCC_NUM_MEL_BINS) + j] = (int8_t)((temp_out[j] / tflite_model_input_scale) + tflite_model_input_zero_point);
        }
        // ESP_LOGI(TAG, "4");
        // assert(heap_caps_check_integrity_all(true));
    }

    // Destroy the model to release internal memory
    // fbank_handle->destroy(fbank_model);
    fbank_handle->clean(fbank_model);
    return 1;
}