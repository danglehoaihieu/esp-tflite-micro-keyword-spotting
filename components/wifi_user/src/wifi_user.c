#include "wifi_user.h"

const char *ssid = "HieuPC_Hotspot";
const char *pass = "123456789";

int retry_num=0;
static bool wifi_connected = false;
static const char *WIFI_USER = "WIFI_USER";

void init_NVS(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND)
    {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
}

void wifi_connection(void)
{
    wifi_connected = false;
    init_NVS();
    // Wi-Fi Configuration Phase
    esp_netif_init();
    esp_event_loop_create_default();     // event loop                    s1.2
    esp_netif_create_default_wifi_sta(); // WiFi station                      s1.3
    wifi_init_config_t wifi_initiation = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&wifi_initiation); //     
    esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event_handler, NULL);
    esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, wifi_event_handler, NULL);
    wifi_config_t wifi_configuration = {.sta = {.ssid = "", .password = "",}};
    strcpy((char*)wifi_configuration.sta.ssid, ssid);
    strcpy((char*)wifi_configuration.sta.password, pass);    
    //esp_log_write(ESP_LOG_INFO, "Kconfig", "SSID=%s, PASS=%s", ssid, pass);
    esp_wifi_set_config(ESP_IF_WIFI_STA, &wifi_configuration);
    esp_wifi_set_mode(WIFI_MODE_STA);
    // 3 - Wi-Fi Start Phase
    esp_wifi_start();
    
    // 4- Wi-Fi Connect Phase
    // esp_wifi_connect();
    // ESP_LOGI(WIFI_USER, "wifi_init_softap finished. SSID:%s  password:%s", ssid, pass);

    while ((! is_wifi_connected()) && (retry_num < WIFI_CONNECT_MAXIMUM_RETRY))
    {
        // printf(".");
        // esp_wifi_connect();
        vTaskDelay(pdMS_TO_TICKS(1000));  // wait 2 seconds before retry
    }
    
    
}
bool is_wifi_connected(void)
{
    return wifi_connected;
}


void wifi_event_handler(void *event_handler_arg, esp_event_base_t event_base, int32_t event_id,void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) 
    {
        esp_wifi_connect();
    } 
    else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED)
    {
        wifi_event_sta_disconnected_t *disconn = (wifi_event_sta_disconnected_t *)event_data;

        ESP_LOGI(WIFI_USER, "Wi-Fi disconnected. Reason: %d", disconn->reason);
        if (retry_num < WIFI_CONNECT_MAXIMUM_RETRY) {
            esp_wifi_connect();
            retry_num++;
            ESP_LOGI(WIFI_USER, "retry to connect to the AP");
        } 
        // ESP_LOGI(WIFI_USER,"connect to the AP fail");
    } 
    else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) 
    {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(WIFI_USER, "Got ip:" IPSTR, IP2STR(&event->ip_info.ip));
        retry_num = 0;
        wifi_connected = true;

    }
}