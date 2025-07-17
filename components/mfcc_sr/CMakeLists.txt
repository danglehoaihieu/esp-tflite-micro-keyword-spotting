idf_component_register(SRCS "./src/mfcc_sr.c"
                    INCLUDE_DIRS "./include/"
                    REQUIRES esp-sr tflite_model)