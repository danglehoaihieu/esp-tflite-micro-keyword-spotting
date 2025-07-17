idf_component_register(SRCS "./src/INMP441.c"
                    INCLUDE_DIRS "./include/"
                    REQUIRES esp_driver_i2s esp_driver_gpio)