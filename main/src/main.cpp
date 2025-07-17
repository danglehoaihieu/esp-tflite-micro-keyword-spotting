#include "main_app.h"


static const char *TAG = "MAIN";
static const char *TIMER_TAG = "TIMER";

static int16_t inmp441_raw[INMP441_SAMPLE_COUNT];
static int16_t inference_buffer[INFERENCE_BUFFER_COUNT];
static int8_t int8_out_features[MFCC_NUM_FRAMES * MFCC_NUM_MEL_BINS];
SemaphoreHandle_t xBinarySemaphore;


// Function that implements the task being created.
void vTaskCode_1( void )
{
    static TickType_t xLastWakeTime = 0;
    // The parameter value is expected to be 1 as 1 is passed in the
    // pvParameters value in the call to xTaskCreateStatic().
    // configASSERT( ( uint32_t ) pvParameters == 1UL );
    ESP_LOGI(TAG, "Task 1 is running");
    while (true)
    {
        if (xTaskGetTickCount() - xLastWakeTime >= FREQUENCY_500MS)
        {
            xLastWakeTime = xTaskGetTickCount();
            // ESP_LOGI(TAG, "Task 1 trigger 500 ms");
        }
        
        // Task code goes here.
    }
    vTaskDelete(NULL);
}

// Function that implements the task being created.
void vTaskCode_2( void )
{
    // The parameter value is expected to be 1 as 1 is passed in the
    // pvParameters value in the call to xTaskCreateStatic().
    // configASSERT( ( uint32_t ) pvParameters == 1UL );
    ESP_LOGI(TAG, "Task 2 is running");
    while (true)
    {
        
        // Task code goes here.
    }
    vTaskDelete(NULL);
}
void RunInferenceTask(void* parameter)
{
    // static TickType_t xLastWakeTime = 0;
    ESP_LOGI(TAG, "RunInferenceTask start ");
    while (true)
    {
        if (xSemaphoreTake(xBinarySemaphore, portMAX_DELAY)) 
        {
            // ESP_LOGI(TAG, "RunInferenceTask start");
            // xLastWakeTime = xTaskGetTickCount();

            // assert(heap_caps_check_integrity_all(true));  // check before
            compute_quantized_int8_mfcc_features(inference_buffer, int8_out_features);
            // assert(heap_caps_check_integrity_all(true));  // check after
            // TickType_t endGetFeature = xTaskGetTickCount();
            model_inference(int8_out_features);
            // TickType_t endModelInference = xTaskGetTickCount();
        
            // ESP_LOGI(TAG, "dps time: %d ms, inference time: %d ms", 
            //     (int)((endGetFeature - xLastWakeTime) * portTICK_PERIOD_MS),
            //     (int)((endModelInference - endGetFeature) * portTICK_PERIOD_MS));
            // ESP_LOGI(TAG, "RunInferenceTask end");
        }
        vTaskDelay(pdMS_TO_TICKS(5));  // Delay to allow other tasks and prevent WDT
    }
    vTaskDelete(NULL);
}
void micTask(void* parameter)
{
    // static TickType_t xLastWakeTime = 0;
    while (true)
    {
        // ESP_LOGI(TAG, "micTask start");
        // xLastWakeTime = xTaskGetTickCount();
        size_t bytesIn = inmp441_read_raw_data(&inmp441_raw[0]);
        // assert(heap_caps_check_integrity_all(true));  // check before

        if (bytesIn > 0) // OK
        {
            websocket_app_transmit((const char*)&inmp441_raw[0], bytesIn);
            #ifndef GET_TRAIN_DATA
            // shift raw data to inference buffer
            // Shift old data to make room
            memmove(inference_buffer,
                &inference_buffer[INMP441_SAMPLE_COUNT],
                (INFERENCE_BUFFER_COUNT - INMP441_SAMPLE_COUNT) * sizeof(int16_t));
            
            // assert(heap_caps_check_integrity_all(true));
            // Copy new data to the end
            memcpy(&inference_buffer[INFERENCE_BUFFER_COUNT - INMP441_SAMPLE_COUNT],
               inmp441_raw,
               INMP441_SAMPLE_BUFF_SIZE);
            
            // assert(heap_caps_check_integrity_all(true));
            #endif
            xSemaphoreGive(xBinarySemaphore);
            // ei_sleep(1000);
            // ESP_LOGI(TAG, "num bytes:%d", sizeof(inmp441_raw));
            // for (int i=0; i < 100; i++)
            // {
            //     printf("%x ", inmp441_raw[i]);
            // }
            // printf("\n");
        }
        // ESP_LOGI(TAG, "t: %d", (int)(xTaskGetTickCount() - xLastWakeTime));
        // ESP_LOGI(TAG, "micTask end");
    }
    vTaskDelete(NULL);
}

void vTask_10ms_Callback( TimerHandle_t pxTimer )
{
    // Optionally do something if the pxTimer parameter is NULL.
    configASSERT( pxTimer );
    ESP_LOGI(TIMER_TAG, "task 10 ms is called");

}
void initalize_timer(void)
{
    TimerHandle_t xTimers;
    xTimers = xTimerCreate( "Task_10ms", (1000 / portTICK_PERIOD_MS), pdTRUE, (void * )0, vTask_10ms_Callback );
    if( xTimers == NULL )
    {
        // The timer was not created.
        ESP_LOGE(TIMER_TAG, "The timer was not created");
    }
    else
    {
        // Start the timer.  No block time is specified, and even if one was
        // it would be ignored because the scheduler has not yet been
        // started.
        if( xTimerStart( xTimers, 1 ) != pdPASS )
        {
            // The timer could not be set into the Active state.
            ESP_LOGE(TIMER_TAG, "The timer was not created");
        }
    }
}
void initialize_task(void)
{
    TaskHandle_t xHandle = NULL;
    xBinarySemaphore = xSemaphoreCreateBinary();
    if (xBinarySemaphore == NULL) {
        ESP_LOGE(TAG, "can not created semaphore");
        return;
    }
    // Create the task, storing the handle.  Note that the passed parameter ucParameterToPass
    // must exist for the lifetime of the task, so in this case is declared static.  If it was just an
    // an automatic stack variable it might no longer exist, or at least have been corrupted, by the time
    // the new task attempts to access it.
    // xTaskCreate( (TaskFunction_t)&vTaskCode_1, "task_1", STACK_SIZE, NULL, tskIDLE_PRIORITY, &xHandle );
    // xTaskCreate( (TaskFunction_t)&vTaskCode_2, "task_2", STACK_SIZE, NULL, tskIDLE_PRIORITY, &xHandle );

    xTaskCreate( (TaskFunction_t)&micTask, "micTask", MIC_STACK_SIZE, NULL, tskIDLE_PRIORITY, &xHandle );
    #ifndef GET_TRAIN_DATA
    xTaskCreate( (TaskFunction_t)&RunInferenceTask, "RunInferenceTask", INFERENCE_STACK_SIZE, NULL, (tskIDLE_PRIORITY), &xHandle );
    #endif
}

void setup(void)
{
    wifi_connection();
    if (is_wifi_connected())
    {
        websocket_app_start();
    }
    inmp441_i2s_init_std();
    mfcc_frontend_init();
    initialize_model();
    // initalize_timer();
    // ...
    // Create tasks here.
    initialize_task();
    // ...
    // vTaskStartScheduler();
    
}
extern "C" void app_main(void)
{
    setup();
}
