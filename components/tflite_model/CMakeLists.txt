idf_component_register(SRCS "./src/tflite_model.cc"
                    INCLUDE_DIRS "./include/"
                    REQUIRES esp-tflite-micro mfcc_sr
                    EMBED_TXTFILES "./model.tflite")