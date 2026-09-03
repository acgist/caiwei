project(caiwei-runtime VERSION 1.0.0 LANGUAGES C CXX)

set(
    CAIWEI_RUNTIME_SRCS
    src/caiwei.cpp
    src/context.cpp
    src/runtime.cpp
    src/session.cpp
    src/context_manager.cpp
    src/session/audio_asr.cpp
    src/session/audio_tts.cpp
    src/session/image_yolo.cpp
    src/session/video_play.cpp
    src/session/video_yolo.cpp
    src/session/embeddings.cpp
    src/session/rerankings.cpp
    src/session/completions.cpp
)
list(
    APPEND CAIWEI_RUNTIME_INCS
    inc
)

if(ENABLE_CAIWEI_RUNTIME_CANN)
    list(
        APPEND CAIWEI_RUNTIME_SRCS
        src/cann/context.cpp
        src/cann/runtime.cpp
        src/cann/yolo_cls.cpp
        src/cann/yolo_det.cpp
        src/cann/yolo_obb.cpp
        src/cann/yolo_seg.cpp
        src/cann/yolo_sem.cpp
        src/cann/yolo_pose.cpp
        src/cann/yolo_depth.cpp
        src/cann/qwen3_asr.cpp
        src/cann/qwen3_tts.cpp
        src/cann/qwen3_llm.cpp
        src/cann/qwen3_vlm.cpp
        src/cann/qwen3_mllm.cpp
        src/cann/qwen3_embedding.cpp
        src/cann/qwen3_reranking.cpp
        src/cann/qwen3_vl_embedding.cpp
        src/cann/qwen3_vl_reranking.cpp
    )
    list(APPEND CAIWEI_RUNTIME_INCS third_party/cann)
endif()

if(ENABLE_CAIWEI_RUNTIME_RKNN)
    list(
        APPEND CAIWEI_RUNTIME_SRCS
        src/rknn/context.cpp
        src/rknn/runtime.cpp
        src/rknn/yolo_cls.cpp
        src/rknn/yolo_det.cpp
        src/rknn/yolo_obb.cpp
        src/rknn/yolo_seg.cpp
        src/rknn/yolo_sem.cpp
        src/rknn/yolo_pose.cpp
        src/rknn/yolo_depth.cpp
        src/rknn/qwen3_asr.cpp
        src/rknn/qwen3_tts.cpp
        src/rknn/qwen3_llm.cpp
        src/rknn/qwen3_vlm.cpp
        src/rknn/qwen3_mllm.cpp
        src/rknn/qwen3_embedding.cpp
        src/rknn/qwen3_reranking.cpp
        src/rknn/qwen3_vl_embedding.cpp
        src/rknn/qwen3_vl_reranking.cpp
    )
    list(APPEND CAIWEI_RUNTIME_INCS third_party/rknn)
endif()

if(ENABLE_CAIWEI_RUNTIME_LLAMACPP)
    list(
        APPEND CAIWEI_RUNTIME_SRCS
        src/llamacpp/context.cpp
        src/llamacpp/runtime.cpp
        src/llamacpp/yolo_cls.cpp
        src/llamacpp/yolo_det.cpp
        src/llamacpp/yolo_obb.cpp
        src/llamacpp/yolo_seg.cpp
        src/llamacpp/yolo_sem.cpp
        src/llamacpp/yolo_pose.cpp
        src/llamacpp/yolo_depth.cpp
        src/llamacpp/qwen3_asr.cpp
        src/llamacpp/qwen3_tts.cpp
        src/llamacpp/qwen3_llm.cpp
        src/llamacpp/qwen3_vlm.cpp
        src/llamacpp/qwen3_mllm.cpp
        src/llamacpp/qwen3_embedding.cpp
        src/llamacpp/qwen3_reranking.cpp
        src/llamacpp/qwen3_vl_embedding.cpp
        src/llamacpp/qwen3_vl_reranking.cpp
    )
endif()

if(ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME)
    list(
        APPEND CAIWEI_RUNTIME_SRCS
        src/onnxruntime/context.cpp
        src/onnxruntime/runtime.cpp
        src/onnxruntime/yolo_cls.cpp
        src/onnxruntime/yolo_det.cpp
        src/onnxruntime/yolo_obb.cpp
        src/onnxruntime/yolo_seg.cpp
        src/onnxruntime/yolo_sem.cpp
        src/onnxruntime/yolo_pose.cpp
        src/onnxruntime/yolo_depth.cpp
        src/onnxruntime/qwen3_asr.cpp
        src/onnxruntime/qwen3_tts.cpp
        src/onnxruntime/qwen3_llm.cpp
        src/onnxruntime/qwen3_vlm.cpp
        src/onnxruntime/qwen3_mllm.cpp
        src/onnxruntime/qwen3_embedding.cpp
        src/onnxruntime/qwen3_reranking.cpp
        src/onnxruntime/qwen3_vl_embedding.cpp
        src/onnxruntime/qwen3_vl_reranking.cpp
    )
endif()

message(STATUS "CAIWEI_RUNTIME_DLLS: ${CAIWEI_RUNTIME_DLLS}")
message(STATUS "CAIWEI_RUNTIME_INCS: ${CAIWEI_RUNTIME_INCS}")
message(STATUS "CAIWEI_RUNTIME_LIBS: ${CAIWEI_RUNTIME_LIBS}")

add_library(
    ${PROJECT_NAME}
    STATIC
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

if(ANDROID)
# -
elseif(CMAKE_HOST_UNIX)
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
