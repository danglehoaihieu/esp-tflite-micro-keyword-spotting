#include "esp_log.h"
#include "esp_websocket_client.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define NO_DATA_TIMEOUT_SEC 10
#ifdef __cplusplus
extern "C" {
#endif
void websocket_app_start(void);
void websocket_app_transmit(const char * data, size_t bytes);
void SetDynamicClientURI(char *ip_str);
bool is_websocket_app_connected(void);

#ifdef __cplusplus
}
#endif
