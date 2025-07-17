#include "tflite_model.h"
#include "tflite_model_shared.h"

const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
int8_t* model_input_buffer = nullptr;
static bool init_model_success = false;
uint8_t *tensor_arena = NULL;
static const char *TAG = "TFLITE";
extern const unsigned char model_data[] asm("_binary_model_tflite_start");


// extern "C" {
    float tflite_model_input_scale;
    int tflite_model_input_zero_point;
// }
void initialize_model(void)
{
    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    tensor_arena = (uint8_t *)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM);

    if (tensor_arena == NULL) {
        ESP_LOGE(TAG, "Malloc failed");
        return;
    }

    model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Model provided is schema version %d not equal to supported "
                    "version %d.",
                    model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    static tflite::MicroMutableOpResolver<15> resolver;
    // Define supported ops
    resolver.AddResizeBilinear();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddConv2D();
    resolver.AddMaxPool2D();


    // Build an interpreter to run the model with.
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        MicroPrintf("AllocateTensors() failed");
        return;
    }

    // Get information about the memory area to use for the model's input.
    input = interpreter->input(0);
    tflite_model_input_scale = input->params.scale;
    tflite_model_input_zero_point = input->params.zero_point;

    ESP_LOGI(TAG, "input->dims->size: %d", input->dims->size);
    ESP_LOGI(TAG, "input->dims: %d, %d, %d, %d", 
        input->dims->data[0], 
        input->dims->data[1],
        input->dims->data[2],
        input->dims->data[3]);
    ESP_LOGI(TAG, "model_input->type: %d", input->type);
    if ((input->dims->size != 4) || (input->dims->data[1] != kFeatureCount) ||
        (input->dims->data[2] != kFeatureSize) ||
        (input->type != kTfLiteInt8)) {
        ESP_LOGE(TAG, "Bad input tensor parameters in model");
        return;
    }

    model_input_buffer = tflite::GetTensorData<int8_t>(input);

    // Keep track of how many inferences we have performed.
    init_model_success = true;
    ESP_LOGI(TAG, "Initialized tflite model successful");
}

int model_inference(int8_t* feature_buffer)
{
    if (! init_model_success) return kTfLiteCancelled;
    // Copy feature buffer to input tensor
    memcpy(model_input_buffer, feature_buffer, kFeatureElementCount);

    // Run the model on the spectrogram input and make sure it succeeds.
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke failed %d", invoke_status);
        return invoke_status;
    }

    // Obtain a pointer to the output tensor
    TfLiteTensor* output = interpreter->output(0);
    int max_idx = 0;
    float max_result = 0.0;
    float sum = 0.0;
    float probs[kCategoryCount];
    // ESP_LOGI(TAG, "output_scale: %f, output_zero_point  %d", output_scale, output_zero_point);


    // Dequantize output values and find the max
    for (int i = 0; i < kCategoryCount; i++) 
    {
        float current_result =
            (tflite::GetTensorData<int8_t>(output)[i] - output->params.zero_point) * output->params.scale;
        probs[i] = exp(current_result);
        sum += probs[i];
        // ESP_LOGI(TAG, "Class %7s score (int8): %d", 
        //             kCategoryLabels[i],
        //             tflite::GetTensorData<int8_t>(output)[i]);
    }

    for (int i = 0; i < kCategoryCount; i++) 
    {
        probs[i] = probs[i]/sum;
        if (probs[i] > max_result) 
        {
            max_result = probs[i]; // update max result
            max_idx = i; // update category
        }

        printf("%7s: %.2f, ", 
                    kCategoryLabels[i],
                    probs[i]);
    }
    printf("\n");

    if (max_result > TFLITE_MODEL_ACCEPT_THRESHOLD) 
    {
        printf("\n");
        ESP_LOGI(TAG, "\tDetected %7s, score: %.2f\n", kCategoryLabels[max_idx],
            static_cast<double>(max_result));
        printf("\n");
    }

    return kTfLiteOk;
}