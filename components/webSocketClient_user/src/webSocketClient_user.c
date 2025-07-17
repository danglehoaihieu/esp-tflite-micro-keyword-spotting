#include "webSocketClient_user.h"

#define WEBSOCKET_CLIENT_PORT   8888
#define WEBSOCKET_CLIENT_URI    "ws://192.168.30.19"
#define WS_URI_MAX_LEN 64

static const char * WEBSOCKET_TAG = "WEB_SOCKET_CLIENT";
static char ws_uri[WS_URI_MAX_LEN];
static esp_websocket_client_handle_t client = NULL;
static void websocket_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data);
// static TimerHandle_t shutdown_signal_timer;
// static SemaphoreHandle_t shutdown_sema;



static void websocket_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data)
{
    esp_websocket_event_data_t *data = (esp_websocket_event_data_t *)event_data;
    switch (event_id) 
    {
        case WEBSOCKET_EVENT_CONNECTED:
            ESP_LOGI(WEBSOCKET_TAG, "WEBSOCKET_EVENT_CONNECTED");
            break;
        case WEBSOCKET_EVENT_DISCONNECTED:
            ESP_LOGI(WEBSOCKET_TAG, "WEBSOCKET_EVENT_DISCONNECTED");
            break;
        case WEBSOCKET_EVENT_DATA:
            // ESP_LOGI(WEBSOCKET_TAG, "WEBSOCKET_EVENT_DATA");
            // ESP_LOGI(WEBSOCKET_TAG, "Received opcode=%d", data->op_code);
            if (data->op_code == 0x08 && data->data_len == 2) 
            {
                // ESP_LOGW(WEBSOCKET_TAG, "Received closed message with code=%d", 256*data->data_ptr[0] + data->data_ptr[1]);
            } 
            else 
            {
                // ESP_LOGW(WEBSOCKET_TAG, "Received=%.*s", data->data_len, (char *)data->data_ptr);
            }
            // ESP_LOGW(WEBSOCKET_TAG, "Total payload length=%d, data_len=%d, current payload offset=%d\r\n", data->payload_len, data->data_len, data->payload_offset);

            // xTimerReset(shutdown_signal_timer, portMAX_DELAY);
            break;
        case WEBSOCKET_EVENT_ERROR:
            ESP_LOGI(WEBSOCKET_TAG, "WEBSOCKET_EVENT_ERROR");
            break;
    }
}
void SetDynamicClientURI(char *ip_str)
{
    snprintf(ws_uri, sizeof(ws_uri), "ws://%s", ip_str);
    ESP_LOGI(WEBSOCKET_TAG, "WebSocket URI: %s", ws_uri);
}

void websocket_app_start(void)
{
    // shutdown_signal_timer = xTimerCreate("Websocket shutdown timer", NO_DATA_TIMEOUT_SEC * 1000 / portTICK_PERIOD_MS,
    //                                      pdFALSE, NULL, shutdown_signaler);
    // shutdown_sema = xSemaphoreCreateBinary();

    esp_websocket_client_config_t websocket_cfg = {
        .task_stack = 8192,  // default 4096
        .uri = WEBSOCKET_CLIENT_URI,
        .port = WEBSOCKET_CLIENT_PORT
    };


    ESP_LOGI(WEBSOCKET_TAG, "Connecting to %s...", websocket_cfg.uri);

    client = esp_websocket_client_init(&websocket_cfg);
    esp_websocket_register_events(client, WEBSOCKET_EVENT_ANY, websocket_event_handler, (void *)client);
            esp_websocket_client_start(client);
}

bool is_websocket_app_connected(void)
{
    return (esp_websocket_client_is_connected(client));
}

void websocket_app_transmit(const char * data, size_t bytes)
{
    if (esp_websocket_client_is_connected(client))
    {
        esp_websocket_client_send_bin(client, data, bytes, portMAX_DELAY);
    }
}
