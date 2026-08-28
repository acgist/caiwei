project(caiwei-runtime VERSION 1.0.0 LANGUAGES C CXX)

set(
    CAIWEI_RUNTIME_SRCS
    src/yolo.cpp
    src/caiwei.cpp
    src/context.cpp
    src/runtime.cpp
    src/session.cpp
    src/session/video_play.cpp
)
list(APPEND CAIWEI_RUNTIME_INCS inc)

if (ENABLE_CAIWEI_RUNTIME_MNN)
    list(
        APPEND CAIWEI_RUNTIME_SRCS
        src/mnn/runtime.cpp
        src/mnn/yolo26.cpp
        src/mnn/qwen3_asr.cpp
        src/mnn/qwen3_tts.cpp
        src/mnn/qwen3_llm.cpp
        src/mnn/qwen3_vlm.cpp
        src/mnn/qwen3_mllm.cpp
        src/mnn/qwen3_embedding.cpp
        src/mnn/qwen3_reranking.cpp
    )
    list(APPEND CAIWEI_RUNTIME_INCS third_party/mnn)
endif()

if (ENABLE_CAIWEI_RUNTIME_CANN)
    list(
        APPEND CAIWEI_RUNTIME_SRCS
        src/cann/runtime.cpp
        src/cann/yolo26.cpp
        src/cann/qwen3_asr.cpp
        src/cann/qwen3_tts.cpp
        src/cann/qwen3_llm.cpp
        src/cann/qwen3_vlm.cpp
        src/cann/qwen3_mllm.cpp
        src/cann/qwen3_embedding.cpp
        src/cann/qwen3_reranking.cpp
    )
    list(APPEND CAIWEI_RUNTIME_INCS third_party/cann)
endif()

if (ENABLE_CAIWEI_RUNTIME_RKNN)
    list(
        APPEND CAIWEI_RUNTIME_SRCS
        src/rknn/runtime.cpp
        src/rknn/yolo26.cpp
        src/rknn/qwen3_asr.cpp
        src/rknn/qwen3_tts.cpp
        src/rknn/qwen3_llm.cpp
        src/rknn/qwen3_vlm.cpp
        src/rknn/qwen3_mllm.cpp
        src/rknn/qwen3_embedding.cpp
        src/rknn/qwen3_reranking.cpp
    )
    list(APPEND CAIWEI_RUNTIME_INCS third_party/rknn)
endif()

if (ENABLE_CAIWEI_RUNTIME_LLAMACPP)
    list(
        APPEND CAIWEI_RUNTIME_SRCS
        src/llamacpp/runtime.cpp
        src/llamacpp/qwen3_asr.cpp
        src/llamacpp/qwen3_tts.cpp
        src/llamacpp/qwen3_llm.cpp
        src/llamacpp/qwen3_vlm.cpp
        src/llamacpp/qwen3_mllm.cpp
        src/llamacpp/qwen3_embedding.cpp
        src/llamacpp/qwen3_reranking.cpp
    )
    list(APPEND CAIWEI_RUNTIME_INCS third_party/llamacpp)
endif()

if (ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME)
    list(
        APPEND CAIWEI_RUNTIME_SRCS
        src/onnxruntime/runtime.cpp
        src/onnxruntime/yolo26.cpp
    )
    list(APPEND CAIWEI_RUNTIME_INCS third_party/onnxruntime)
endif()

message(STATUS "CAIWEI_RUNTIME_DLLS: ${CAIWEI_RUNTIME_DLLS}")
message(STATUS "CAIWEI_RUNTIME_INCS: ${CAIWEI_RUNTIME_INCS}")
message(STATUS "CAIWEI_RUNTIME_LIBS: ${CAIWEI_RUNTIME_LIBS}")

add_library(
    ${PROJECT_NAME}
    ${CAIWEI_RUNTIME_SRCS}
)
target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC
    caiwei-media
    ${CAIWEI_RUNTIME_LIBS}
)
target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
    ${CAIWEI_RUNTIME_INCS}
)

if (ANDROID)
# -
elseif (CMAKE_HOST_UNIX)
    install(
        TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION bin
        RUNTIME DESTINATION bin
        PUBLIC_HEADER DESTINATION inc
    )
elseif(CMAKE_HOST_WIN32)
# -
else()
# -
endif()
