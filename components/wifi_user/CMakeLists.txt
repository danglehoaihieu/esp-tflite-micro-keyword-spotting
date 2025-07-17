idf_component_register(SRCS "./src/wifi_user.c"
                    INCLUDE_DIRS "./include/"
                    REQUIRES esp_wifi nvs_flash)