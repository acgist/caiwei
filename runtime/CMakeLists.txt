project(caiwei-runtime VERSION 1.0.0 LANGUAGES C CXX)

set(
    CAIWEI_RUNTIME_SRCS
    src/caiwei.cpp
    src/context.cpp
    src/runtime.cpp
    src/session.cpp
    src/context_manager.cpp
    src/session/video_play.cpp
    src/session/video_yolo.cpp
    src/session/embeddings.cpp
    src/session/rerankings.cpp
    src/session/completions.cpp
)

list(
    APPEND CAIWEI_RUNTIME_INCS
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

list(
    APPEND CAIWEI_RUNTIME_LIBS
    caiwei-media
)

if(ENABLE_CAIWEI_RUNTIME_CANN)
    list(
        APPEND CAIWEI_RUNTIME_SRCS
        src/cann/context.cpp
        src/cann/runtime.cpp
        src/cann/yolo_cls.cpp
        src/cann/yolo_det.cpp
        src/cann/yolo_seg.cpp
        src/cann/yolo_pose.cpp
        src/cann/qwen3_asr.cpp
        src/cann/qwen3_llm.cpp
        src/cann/qwen3_vlm.cpp
        src/cann/qwen3_embedding.cpp
        src/cann/qwen3_reranking.cpp
    )
endif()

if(ENABLE_CAIWEI_RUNTIME_RKNN2)
    list(
        APPEND CAIWEI_RUNTIME_SRCS
        src/rknn2/context.cpp
        src/rknn2/runtime.cpp
        src/rknn2/yolo_cls.cpp
        src/rknn2/yolo_det.cpp
        src/rknn2/yolo_seg.cpp
        src/rknn2/yolo_pose.cpp
        src/rknn2/qwen3_asr.cpp
        src/rknn2/qwen3_llm.cpp
        src/rknn2/qwen3_vlm.cpp
        src/rknn2/qwen3_embedding.cpp
        src/rknn2/qwen3_reranking.cpp
    )
endif()

if(ENABLE_CAIWEI_RUNTIME_RKNN3)
    list(
        APPEND CAIWEI_RUNTIME_SRCS
        src/rknn3/context.cpp
        src/rknn3/runtime.cpp
        src/rknn3/yolo_cls.cpp
        src/rknn3/yolo_det.cpp
        src/rknn3/yolo_seg.cpp
        src/rknn3/yolo_pose.cpp
        src/rknn3/qwen3_asr.cpp
        src/rknn3/qwen3_llm.cpp
        src/rknn3/qwen3_vlm.cpp
        src/rknn3/qwen3_embedding.cpp
        src/rknn3/qwen3_reranking.cpp
    )
endif()

if(ENABLE_CAIWEI_RUNTIME_LLAMACPP)
    list(
        APPEND CAIWEI_RUNTIME_SRCS
        src/llamacpp/context.cpp
        src/llamacpp/runtime.cpp
        src/llamacpp/yolo_cls.cpp
        src/llamacpp/yolo_det.cpp
        src/llamacpp/yolo_seg.cpp
        src/llamacpp/yolo_pose.cpp
        src/llamacpp/qwen3_asr.cpp
        src/llamacpp/qwen3_llm.cpp
        src/llamacpp/qwen3_vlm.cpp
        src/llamacpp/qwen3_embedding.cpp
        src/llamacpp/qwen3_reranking.cpp
    )
endif()

if(ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME)
    list(
        APPEND CAIWEI_RUNTIME_SRCS
        src/onnxruntime/context.cpp
        src/onnxruntime/runtime.cpp
        src/onnxruntime/yolo_cls.cpp
        src/onnxruntime/yolo_det.cpp
        src/onnxruntime/yolo_seg.cpp
        src/onnxruntime/yolo_pose.cpp
        src/onnxruntime/qwen3_asr.cpp
        src/onnxruntime/qwen3_llm.cpp
        src/onnxruntime/qwen3_vlm.cpp
        src/onnxruntime/qwen3_embedding.cpp
        src/onnxruntime/qwen3_reranking.cpp
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
        TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        PUBLIC_HEADER DESTINATION include
    )
elseif(CMAKE_HOST_WIN32)
# -
else()
# -
endif()
