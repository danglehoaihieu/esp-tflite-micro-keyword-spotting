#include <stdio.h>

#define I2S_STD
#ifdef I2S_STD
    #include "driver/i2s_std.h"
    #include "driver/i2s_pdm.h"
    #define PIN_DOUT    I2S_GPIO_UNUSED
#else 
    #include "driver/i2s.h"   
#endif

#include "driver/gpio.h"
// #include "freertos/FreeRTOS.h"

// #define GET_TRAIN_DATA

#ifdef GET_TRAIN_DATA
    #define INMP441_SAMPLE_COUNT 1000
#endif
#ifndef GET_TRAIN_DATA
    #define INMP441_SAMPLE_COUNT 1000*2
#endif
#define INMP441_SAMPLE_BUFF_SIZE (INMP441_SAMPLE_COUNT * sizeof(int16_t))

#define DMA_CNT                          10

#define PIN_BCLK    GPIO_NUM_4
#define PIN_WS      GPIO_NUM_5

#define PIN_DIN     GPIO_NUM_18
#define I2S_PORT    I2S_NUM_0

void run_module_1(void);
void call_to_module_1(void);
void inmp441_i2s_init_pdm(void);
void inmp441_i2s_init_lagacy(void);
#ifdef __cplusplus
extern "C" {
#endif
void inmp441_i2s_init_std(void);
size_t inmp441_read_raw_data(int16_t * desc_buf);
#ifdef __cplusplus
}
#endif
esp_err_t inmp441_read_raw_data_pdm(int16_t * desc_buf);
esp_err_t inmp441_read_raw_data_legacy(int16_t * desc_buf);