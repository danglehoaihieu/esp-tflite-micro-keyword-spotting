# set(TFLM_THIRD_PARTY_FOLDER ../managed_components/espressif__esp-tflite-micro/third_party)

# set(KISS_FFT_FOLDER ${TFLM_THIRD_PARTY_FOLDER}/kissfft)
# set(KISS_FFT_TOOL_FOLDER ${KISS_FFT_FOLDER}/tools)

# file(GLOB_RECURSE KISS_FFT_SRC_FILES ${KISS_FFT_FOLDER}/*.c)
# file(GLOB_RECURSE KISS_FFT_TOOL_SRC_FILES ${KISS_FFT_TOOL_FOLDER}/*.c)

# list(APPEND SRC_FILES ${KISS_FFT_SRC_FILES})
# list(APPEND SRC_FILES ${KISS_FFT_TOOL_SRC_FILES})

# set(include_dirs
#     ${KISS_FFT_FOLDER}
#     ${KISS_FFT_TOOL_FOLDER}
# )

idf_component_register(SRCS "./src/main.cpp" #"${SRC_FILES}"
                    INCLUDE_DIRS "./include/" #"${include_dirs}"
                    )
