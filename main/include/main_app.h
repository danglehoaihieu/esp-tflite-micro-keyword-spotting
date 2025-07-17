#include <stdio.h>

#include "esp_log.h"

#include "INMP441.h"
#include "wifi_user.h"
#include "webSocketClient_user.h"
#include "tflite_model.h"
// #include "mfcc_user.h"
#include "mfcc_sr.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
// #include "inference_buffer.h"

// Dimensions of the buffer that the task being created will use as its stack.
// NOTE:  This is the number of words the stack will hold, not the number of
// bytes.  For example, if each stack item is 32-bits, and this is set to 100,
// then 400 bytes (100 * 32-bits) will be allocated.
#define STACK_SIZE 8 * 1024 // 200
#define INFERENCE_STACK_SIZE (20 * 1024 / sizeof(StackType_t))
#define MIC_STACK_SIZE 20 * 1024
#define FREQUENCY_500MS 500
#define INFERENCE_BUFFER_COUNT   MFCC_SAMPLE_RATE * MFCC_AUDIO_DURATION_MS / 1000

void initialize_task(void);
void initialize_timer(void);
void setup(void);
void vTaskCode(void);
void vTaskCode_1( void );
void vTaskCode_2( void );
void vTask_1ms_Callback( TimerHandle_t pxTimer );

