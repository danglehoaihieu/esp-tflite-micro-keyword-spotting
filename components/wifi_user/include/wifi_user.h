#include <stdio.h> //for basic printf commands
#include <string.h> //for handling strings
#include "esp_system.h" //esp_init funtions esp_err_t 
#include "esp_wifi.h" //esp_wifi_init functions and wifi operations
#include "esp_log.h" //for showing logs
#include "esp_event.h" //for wifi event
#include "nvs_flash.h" //non volatile storage
#include "lwip/err.h" //light weight ip packets error handling
#include "lwip/sys.h" //system applications for light weight ip apps
#include <stdbool.h>
#define WIFI_CONNECT_MAXIMUM_RETRY 5


void init_NVS(void);
#ifdef __cplusplus
extern "C" {
#endif
void wifi_connection(void);
bool is_wifi_connected(void);
#ifdef __cplusplus
}
#endif
void wifi_event_handler(void *event_handler_arg, esp_event_base_t event_base, int32_t event_id,void *event_data);