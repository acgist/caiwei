cmake_minimum_required(VERSION 3.22.0)

project(caiwei-project VERSION 1.0.0 LANGUAGES C CXX)

# https://gcc.gnu.org/projects/c-status.html
# https://gcc.gnu.org/projects/cxx-status.html
# https://gcc.gnu.org/onlinedocs/gcc/Standards.html
# https://gcc.gnu.org/onlinedocs/gcc/Option-Summary.html
# https://cmake.org/cmake/help/latest/prop_tgt/C_STANDARD.html
# https://cmake.org/cmake/help/latest/prop_tgt/CXX_STANDARD.html
# https://learn.microsoft.com/zh-cn/cpp/build/reference/compiler-options-listed-by-category

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_CAIWEI_TEST                "Build Test"     OFF) # 编译测试
option(ENABLE_CAIWEI_PERF                "Performance"    ON)  # 性能统计
option(ENABLE_CAIWEI_BACKEND_CUDA        "Nvidia GPU"     OFF) # llama.cpp ONNXRuntime
option(ENABLE_CAIWEI_RUNTIME_RKNN2       "Rockchip RKNN2" OFF) # RKNN2
option(ENABLE_CAIWEI_RUNTIME_RKNN3       "Rockchip RKNN3" OFF) # RKNN3
option(ENABLE_CAIWEI_RUNTIME_LLAMACPP    "llama.cpp"      ON)  # llama.cpp
option(ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME "ONNXRuntime"    ON)  # ONNXRuntime

# TODO 升级 23
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_HOST_UNIX)
    add_definitions(-DCAIWEI_OS_UNIX=1)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E time")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
elseif(CMAKE_HOST_WIN32)
    if(MSVC)
        add_definitions(-DCAIWEI_OS_WIN=1)
        add_compile_options(
            $<$<COMPILE_LANGUAGE:C>:/utf-8>
            $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus>
        )
        add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    else()
        message(FATAL_ERROR "没有安装MSVC开发环境")
    endif()
else()
    message(FATAL_ERROR "不支持的操作系统: ${CMAKE_HOST_SYSTEM_NAME}")
endif()

get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT is_multi_config)
    if(NOT CMAKE_BUILD_TYPE)
        message(STATUS "设置默认编译模式: Release")
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    endif()
    message(STATUS "编译模式: ${CMAKE_BUILD_TYPE}")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<IF:$<CONFIG:Debug>,Debug,Release>/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<IF:$<CONFIG:Debug>,Debug,Release>/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<IF:$<CONFIG:Debug>,Debug,Release>/bin")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug/bin)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release/bin)

message(STATUS "CMAKE_HOST_SYSTEM_NAME: ${CMAKE_HOST_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_C_FLAGS:           ${CMAKE_C_FLAGS}")
message(STATUS "CMAKE_C_FLAGS_DEBUG:     ${CMAKE_C_FLAGS_DEBUG}")
message(STATUS "CMAKE_C_FLAGS_RELEASE:   ${CMAKE_C_FLAGS_RELEASE}")
message(STATUS "CMAKE_CXX_FLAGS:         ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG:   ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")

set(CAIWEI_AUDIO_DECODER "" CACHE STRING "优先使用的音频解码器名: aac/mp3/opus")
set(CAIWEI_VIDEO_DECODER "" CACHE STRING "优先使用的视频解码器名: h264_qsv/h264_cuvid/h264_rkmpp/hevc_qsv/hevc_cuvid/hevc_rkmpp")
set(CAIWEI_AUDIO_ENCODER "" CACHE STRING "优先使用的音频编码器名: aac/mp3/opus")
set(CAIWEI_VIDEO_ENCODER "" CACHE STRING "优先使用的视频编码器名: h264_qsv/h264_cuvid/h264_rkmpp/hevc_qsv/hevc_cuvid/hevc_rkmpp")

if(NOT CAIWEI_AUDIO_DECODER STREQUAL "")
    message(STATUS "Enable audio decoder: ${CAIWEI_AUDIO_DECODER}")
    add_compile_definitions(CAIWEI_AUDIO_DECODER="${CAIWEI_AUDIO_DECODER}")
endif()
if(NOT CAIWEI_VIDEO_DECODER STREQUAL "")
    message(STATUS "Enable video decoder: ${CAIWEI_VIDEO_DECODER}")
    add_compile_definitions(CAIWEI_VIDEO_DECODER="${CAIWEI_VIDEO_DECODER}")
endif()
if(NOT CAIWEI_AUDIO_ENCODER STREQUAL "")
    message(STATUS "Enable audio encoder: ${CAIWEI_AUDIO_ENCODER}")
    add_compile_definitions(CAIWEI_AUDIO_ENCODER="${CAIWEI_AUDIO_ENCODER}")
endif()
if(NOT CAIWEI_VIDEO_ENCODER STREQUAL "")
    message(STATUS "Enable video encoder: ${CAIWEI_VIDEO_ENCODER}")
    add_compile_definitions(CAIWEI_VIDEO_ENCODER="${CAIWEI_VIDEO_ENCODER}")
endif()

if(ENABLE_CAIWEI_TEST)
    message(STATUS "编译测试")
    add_definitions(-DENABLE_CAIWEI_TEST=1)
endif()

if(ENABLE_CAIWEI_PERF)
    message(STATUS "性能统计")
    add_definitions(-DENABLE_CAIWEI_PERF=1)
endif()

if(ENABLE_CAIWEI_BACKEND_CUDA)
    message(STATUS "Enable Nvidia GPU")
    add_definitions(-DENABLE_CAIWEI_BACKEND_CUDA=1)
    if(CMAKE_HOST_UNIX)
        # 指定CUDA量化参数减少编译时间
        set(GGML_CUDA_FA_QUANTS "q4_0-q4_0,q8_0-q8_0,f16-f16,bf16-bf16" CACHE STRING "CUDA量化参数")
        # 指定CUDA架构列表减少编译时间
        set(CMAKE_CUDA_ARCHITECTURES 70 80 89 90 CACHE STRING "CUDA架构列表")
    elseif(CMAKE_HOST_WIN32)
        # 指定CUDA量化参数减少编译时间
        set(GGML_CUDA_FA_QUANTS "q4_0-q4_0,q8_0-q8_0" CACHE STRING "CUDA量化参数")
        # 指定CUDA架构列表减少编译时间
        set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA架构列表")
    else()
        message(FATAL_ERROR "CUDA不支持的平台")
    endif()
    find_package(CUDAToolkit REQUIRED)
endif()

if(ENABLE_CAIWEI_RUNTIME_RKNN2)
    message(STATUS "Enable Rockchip RKNN2")
    add_definitions(-DENABLE_CAIWEI_RUNTIME_RKNN2=1)
    set(ENABLE_CAIWEI_BACKEND_RKNN ON)
    if(CMAKE_HOST_UNIX)
        list(APPEND CAIWEI_RUNTIME_LIBS rknnrt)
    elseif(CMAKE_HOST_WIN32)
        message(FATAL_ERROR "RKNN2不支持Windows平台")
    else()
        message(FATAL_ERROR "RKNN2不支持的平台")
    endif()
endif()

if(ENABLE_CAIWEI_RUNTIME_RKNN3)
    message(STATUS "Enable Rockchip RKNN3")
    add_definitions(-DENABLE_CAIWEI_RUNTIME_RKNN3=1)
    set(ENABLE_CAIWEI_BACKEND_RKNN ON)
    if(CMAKE_HOST_UNIX)
        list(APPEND CAIWEI_RUNTIME_LIBS rknn3_api)
    elseif(CMAKE_HOST_WIN32)
        message(FATAL_ERROR "RKNN3不支持Windows平台")
    else()
        message(FATAL_ERROR "RKNN3不支持的平台")
    endif()
endif()

if(ENABLE_CAIWEI_BACKEND_RKNN)
    message(STATUS "Enable Rockchip NPU")
    add_definitions(-DENABLE_CAIWEI_BACKEND_RKNN=1)
    if(CMAKE_HOST_UNIX)
        list(APPEND CAIWEI_RUNTIME_LIBS rga)
    elseif(CMAKE_HOST_WIN32)
        message(FATAL_ERROR "RKNN不支持Windows平台")
    else()
        message(FATAL_ERROR "RKNN不支持的平台")
    endif()
endif()

if(ENABLE_CAIWEI_RUNTIME_LLAMACPP)
    message(STATUS "Enable llama.cpp")
    add_definitions(-DENABLE_CAIWEI_RUNTIME_LLAMACPP=1)
    set(LLAMACPP_PATH "${CMAKE_SOURCE_DIR}/deps/llama.cpp" CACHE STRING "llama.cpp Path")
    if(ENABLE_CAIWEI_BACKEND_CUDA)
        set(GGML_CUDA ON)
    endif()
    set(LLAMA_BUILD_MTMD ON)
    add_subdirectory("${LLAMACPP_PATH}")
    set_target_properties(ggml ggml-cpu ggml-base mtmd llama vendor-hash PROPERTIES FOLDER "llama.cpp")
    list(APPEND CAIWEI_RUNTIME_LIBS ggml ggml-cpu ggml-base mtmd llama vendor-hash)
    if(ENABLE_CAIWEI_BACKEND_CUDA)
        set_target_properties(ggml-cuda PROPERTIES FOLDER "llama.cpp")
        list(APPEND CAIWEI_RUNTIME_LIBS ggml-cuda)
    endif()
endif()

if(ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME)
    message(STATUS "Enable ONNXRuntime")
    add_definitions(-DENABLE_CAIWEI_RUNTIME_ONNXRUNTIME=1)
    set(ONNXRUNTIME_PATH "${CMAKE_SOURCE_DIR}/deps/onnxruntime" CACHE STRING "ONNXRuntime Path")
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${ONNXRUNTIME_PATH}")
    if(CMAKE_HOST_UNIX)
        file(GLOB ONNXRUNTIME_LIBS "${ONNXRUNTIME_PATH}/lib/libonnxruntime.so")
        list(APPEND CAIWEI_RUNTIME_INCS "${ONNXRUNTIME_PATH}/include")
    elseif(CMAKE_HOST_WIN32)
        file(GLOB ONNXRUNTIME_DLLS "${ONNXRUNTIME_PATH}/lib/*.dll")
        file(GLOB ONNXRUNTIME_LIBS "${ONNXRUNTIME_PATH}/lib/*.lib")
        list(APPEND CAIWEI_SERVER_DLLS ${ONNXRUNTIME_DLLS})
        list(APPEND CAIWEI_RUNTIME_INCS "${ONNXRUNTIME_PATH}/include")
    else()
        message(FATAL_ERROR "ONNXRuntime不支持的平台")
    endif()
    list(APPEND CAIWEI_RUNTIME_LIBS ${ONNXRUNTIME_LIBS})
endif()

if(CMAKE_HOST_UNIX)
    set(FFMPEG_PATH "" CACHE STRING "FFmpeg Path")
    if(FFMPEG_PATH STREQUAL "")
        list(APPEND CAIWEI_RUNTIME_LIBS avutil avcodec avdevice avformat swscale swresample)
    else()
        list(
            APPEND CAIWEI_RUNTIME_INCS
            "${FFMPEG_PATH}/include/"
        )
        list(
            APPEND CAIWEI_RUNTIME_LIBS
            "${FFMPEG_PATH}/lib/libavutil.so"
            "${FFMPEG_PATH}/lib/libavcodec.so"
            "${FFMPEG_PATH}/lib/libavdevice.so"
            "${FFMPEG_PATH}/lib/libavformat.so"
            "${FFMPEG_PATH}/lib/libswscale.so"
            "${FFMPEG_PATH}/lib/libswresample.so"
        )
    endif()
elseif(CMAKE_HOST_WIN32)
    set(FFMPEG_PATH "${CMAKE_SOURCE_DIR}/deps/ffmpeg" CACHE STRING "FFmpeg Path")
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${FFMPEG_PATH}")
    find_package(FFMPEG REQUIRED)
    file(GLOB FFMPEG_DLLS_DEBUG "${FFMPEG_PATH}/debug/bin/*.dll")
    file(GLOB FFMPEG_DLLS_RELEASE "${FFMPEG_PATH}/bin/*.dll")
    list(APPEND CAIWEI_SERVER_DLLS_DEBUG ${FFMPEG_DLLS_DEBUG})
    list(APPEND CAIWEI_SERVER_DLLS_RELEASE ${FFMPEG_DLLS_RELEASE})
    list(APPEND CAIWEI_RUNTIME_INCS ${FFMPEG_INCLUDE_DIRS})
    list(APPEND CAIWEI_RUNTIME_LIBS ${FFMPEG_LIBRARIES})
else()
    message(FATAL_ERROR "FFmpeg不支持的平台")
endif()

if(CMAKE_HOST_UNIX)
    find_file(HTTPLIB_H NAMES httplib.h)
    if(NOT HTTPLIB_H)
        message(FATAL_ERROR "没有安装cpp-httplib")
    endif()
elseif(CMAKE_HOST_WIN32)
    set(CPP_HTTPLIB_PATH "${CMAKE_SOURCE_DIR}/deps/cpp-httplib" CACHE STRING "cpp-httplib Path")
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${CPP_HTTPLIB_PATH}")
    find_package(httplib REQUIRED)
    file(GLOB CPP_HTTPLIB_DLLS_DEBUG "${CPP_HTTPLIB_PATH}/debug/bin/*.dll")
    file(GLOB CPP_HTTPLIB_DLLS_RELEASE "${CPP_HTTPLIB_PATH}/bin/*.dll")
    list(APPEND CAIWEI_SERVER_DLLS_DEBUG ${CPP_HTTPLIB_DLLS_DEBUG})
    list(APPEND CAIWEI_SERVER_DLLS_RELEASE ${CPP_HTTPLIB_DLLS_RELEASE})
    list(APPEND CAIWEI_RUNTIME_LIBS $<IF:$<TARGET_EXISTS:cpp-httplib>,cpp-httplib,httplib::httplib>)
else()
    message(FATAL_ERROR "cpp-httplib不支持的平台")
endif()

add_compile_definitions(
    CAIWEI_DEBUG=$<IF:$<CONFIG:Debug>,1,0>
    CAIWEI_LOG_LEVEL=$<IF:$<CONFIG:Debug>,D,I>
)

add_subdirectory(runtime)
add_subdirectory(server)
if(ENABLE_CAIWEI_TEST)
    add_subdirectory(test)
endif()
