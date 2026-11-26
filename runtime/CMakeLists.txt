project(caiwei-runtime VERSION 1.0.0)

set(
    CAIWEI_RUNTIME_SRCS
    src/env.cpp
    src/log.cpp
    src/caiwei.cpp
    src/context.cpp
    src/runtime.cpp
    src/session.cpp
    src/manager.cpp
    src/tokenizer.cpp
    src/media/text_data.cpp
    src/media/text_tool.cpp
    src/media/media_data.cpp
    src/media/media_tool.cpp
    src/media/media_muxer.cpp
    src/media/media_format.cpp
    src/media/media_demuxer.cpp
    src/session/video_play.cpp
    src/session/embeddings.cpp
    src/session/rerankings.cpp
    src/session/completions.cpp
)

list(
    APPEND CAIWEI_RUNTIME_INCS
    "${CMAKE_CURRENT_SOURCE_DIR}/inc"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
)

if(ENABLE_CAIWEI_RUNTIME_RKNN2)
    list(
        APPEND CAIWEI_RUNTIME_SRCS
        src/rknn2/context.cpp
        src/rknn2/runtime.cpp
        src/rknn2/yolo_cls.cpp
        src/rknn2/yolo_det.cpp
    )
endif()

if(ENABLE_CAIWEI_RUNTIME_RKNN3)
    list(
        APPEND CAIWEI_RUNTIME_SRCS
        src/rknn3/context.cpp
        src/rknn3/runtime.cpp
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
    )
endif()

message(STATUS "CAIWEI_RUNTIME_INCS: ${CAIWEI_RUNTIME_INCS}")
message(STATUS "CAIWEI_RUNTIME_LIBS: ${CAIWEI_RUNTIME_LIBS}")

add_library(${PROJECT_NAME} STATIC ${CAIWEI_RUNTIME_SRCS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${CAIWEI_RUNTIME_LIBS})
target_include_directories(${PROJECT_NAME} PUBLIC ${CAIWEI_RUNTIME_INCS})

if(CMAKE_HOST_UNIX)
    target_compile_options(${PROJECT_NAME} PUBLIC -Wall -Wextra)
elseif(CMAKE_HOST_WIN32)
    target_compile_options(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:C>:/W2> $<$<COMPILE_LANGUAGE:CXX>:/W2>)
else()
    # -
endif()

if(CMAKE_HOST_UNIX)
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-mavx2" CXX_HAS_MAVX2)
        check_cxx_compiler_flag("-mf16c" CXX_HAS_MF16C)
        if(CXX_HAS_MAVX2 AND CXX_HAS_MF16C)
            target_compile_options(${PROJECT_NAME} PUBLIC -mavx2 -mf16c)
        else()
            message(FATAL_ERROR "AVX2/F16C not supported")
        endif()
    endif()
elseif(CMAKE_HOST_WIN32)
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64")
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("/arch:AVX2" CXX_HAS_MAVX2)
        if(CXX_HAS_MAVX2)
            target_compile_options(${PROJECT_NAME} PUBLIC /arch:AVX2)
            target_compile_definitions(${PROJECT_NAME} PUBLIC HAS_AVX2)
            target_compile_definitions(${PROJECT_NAME} PUBLIC HAS_F16C)
        else()
            message(FATAL_ERROR "AVX2/F16C not supported")
        endif()
    endif()
else()
    # -
endif()
