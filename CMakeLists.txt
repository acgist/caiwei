cmake_minimum_required(VERSION 3.22.0)

project(caiwei-project VERSION 1.0.0 LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_CAIWEI_TEST                "Build Test"        OFF) # 是否编译测试
option(ENABLE_CAIWEI_BACKEND_CANN        "Ascend NPU"        OFF) #          llama.cpp ONNXRuntime
option(ENABLE_CAIWEI_BACKEND_RKNN        "Rockchip NPU"      OFF) #                    ONNXRuntime
option(ENABLE_CAIWEI_BACKEND_CUDA        "Nvidia GPU"        OFF) # MNN      llama.cpp ONNXRuntime
option(ENABLE_CAIWEI_BACKEND_VULKAN      "Vulkan GPU"        OFF) # MNN NCNN llama.cpp
option(ENABLE_CAIWEI_BACKEND_OPENVINO    "Intel CPU/GPU/NPU" OFF) #          llama.cpp ONNXRuntime
option(ENABLE_CAIWEI_BACKEND_TENSORRT    "Nvidia TensorRT"   OFF) #                    ONNXRuntime
option(ENABLE_CAIWEI_RUNTIME_CANN        "CANN"              OFF) # 昇腾
option(ENABLE_CAIWEI_RUNTIME_RKNN        "RKNN"              OFF) # 瑞芯微
option(ENABLE_CAIWEI_RUNTIME_MNN         "MNN"               OFF) # 跨平台
option(ENABLE_CAIWEI_RUNTIME_NCNN        "NCNN"              OFF) # 跨平台
option(ENABLE_CAIWEI_RUNTIME_LLAMACPP    "llama.cpp"         ON)  # 跨平台
option(ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME "ONNXRuntime"       ON)  # 跨平台
option(ENABLE_CAIWEI_ASR_ANS             "ASR ANS"           ON)  # ASR前置ANS
option(ENABLE_CAIWEI_ASR_VAD             "ASR VAD"           ON)  # ASR前置VAD
option(ENABLE_CAIWEI_ASR_AGC             "ASR AGC"           ON)  # ASR前置AGC
option(ENABLE_CAIWEI_YOLO_I8             "YOLO INT8"         ON)  # YOLO INT8量化
option(ENABLE_CAIWEI_YOLO_E2E            "YOLO end2end"      ON)  # YOLO end2end模式
option(ENABLE_CAIWEI_SDL2                "SDL2 Player"       ON)  # SDL2播放器
option(ENABLE_CAIWEI_FFMPEG              "FFmpeg"            ON)  # FFmpeg
option(ENABLE_CAIWEI_CPPHTTPLIB          "cpp-httplib"       ON)  # cpp-httplib

if(ANDROID)
    set(CMAKE_C_STANDARD 17)
    set(CMAKE_C_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    add_definitions(-DOS_ANDROID=1)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
elseif(CMAKE_HOST_UNIX)
    set(CMAKE_C_STANDARD 17)
    set(CMAKE_C_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    add_definitions(-DOS_LINUX=1)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E time")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
elseif(CMAKE_HOST_WIN32)
    if(MSVC)
        set(CMAKE_C_STANDARD 17)
        set(CMAKE_C_STANDARD_REQUIRED ON)
        set(CMAKE_CXX_STANDARD 20)
        set(CMAKE_CXX_STANDARD_REQUIRED ON)
        set(CMAKE_CXX_EXTENSIONS OFF)
        add_definitions(-DOS_WIN=1)
        add_compile_options("/utf-8" "/Zc:__cplusplus")
    else()
        message(FATAL_ERROR "需要安装MSVC开发环境")
    endif()
else()
    message(FATAL_ERROR "不支持的系统：${CMAKE_HOST_SYSTEM_NAME}")
endif()

if (ENABLE_ASAN)
	message(STATUS "Build with ASAN")
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DCAIWEI_LOG_DEBUG=D)
    message(STATUS "C_FLAGS: ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG}")
    message(STATUS "CXX_FLAGS: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
else()
    add_definitions(-DCAIWEI_LOG_DEBUG=I)
    message(STATUS "C_FLAGS: ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE}")
    message(STATUS "CXX_FLAGS: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
endif()

if (ENABLE_CAIWEI_BACKEND_CANN)
    message(STATUS "Enable Ascend NPU")
    add_definitions(-DENABLE_CAIWEI_BACKEND_CANN)
endif()

if (ENABLE_CAIWEI_BACKEND_RKNN)
    message(STATUS "Enable Rockchip NPU")
    add_definitions(-DENABLE_CAIWEI_BACKEND_RKNN)
endif()

if (ENABLE_CAIWEI_BACKEND_CUDA)
    message(STATUS "Enable Nvidia GPU")
    add_definitions(-DENABLE_CAIWEI_BACKEND_CUDA)
endif()

if (ENABLE_CAIWEI_BACKEND_VULKAN)
    message(STATUS "Enable Vulkan GPU")
    add_definitions(-DENABLE_CAIWEI_BACKEND_VULKAN)
endif()

if (ENABLE_CAIWEI_BACKEND_OPENVINO)
    message(STATUS "Enable Intel CPU/GPU/NPU")
    add_definitions(-DENABLE_CAIWEI_BACKEND_OPENVINO)
    # CPU GPU NPU
    add_definitions(-DOPENVINO_BACKEND=CPU)
endif()

if (ENABLE_CAIWEI_BACKEND_TENSORRT)
    message(STATUS "Enable Nvidia TensorRT")
    add_definitions(-DENABLE_CAIWEI_BACKEND_TENSORRT)
endif()

if (ENABLE_CAIWEI_RUNTIME_MNN)
    message(STATUS "Running with MNN")
    add_definitions(-DENABLE_CAIWEI_RUNTIME_MNN)
endif()

if (ENABLE_CAIWEI_RUNTIME_NCNN)
    message(STATUS "Running with NCNN")
    add_definitions(-DENABLE_CAIWEI_RUNTIME_NCNN)
endif()

if (ENABLE_CAIWEI_RUNTIME_CANN)
    message(STATUS "Running with CANN")
    add_definitions(-DENABLE_CAIWEI_RUNTIME_CANN)
endif()

if (ENABLE_CAIWEI_RUNTIME_RKNN)
    message(STATUS "Running with RKNN")
    add_definitions(-DENABLE_CAIWEI_RUNTIME_RKNN)
    list(APPEND CAIWEI_RUNTIME_LIBS rknnrt)
endif()

if (ENABLE_CAIWEI_RUNTIME_LLAMACPP)
    message(STATUS "Running with llama.cpp")
    add_definitions(-DENABLE_CAIWEI_RUNTIME_LLAMACPP)
endif()

if (ENABLE_CAIWEI_RUNTIME_ONNXRUNTIME)
    message(STATUS "Running with ONNXRuntime")
    add_definitions(-DENABLE_CAIWEI_RUNTIME_ONNXRUNTIME)
    set(ONNXRUNTIME_PATH "deps/onnxruntime" CACHE STRING "ONNXRuntime Path")
    set(
        CMAKE_PREFIX_PATH
        ${CMAKE_PREFIX_PATH}
        ${ONNXRUNTIME_PATH}
    )
    if (ANDROID)
    # -
    elseif(CMAKE_HOST_UNIX)
        find_package(onnxruntime REQUIRED)
        set(ONNXRUNTIME_LIBS onnxruntime::onnxruntime)
    elseif(CMAKE_HOST_WIN32)
        file(GLOB ONNXRUNTIME_DLLS "${ONNXRUNTIME_PATH}/lib/*.dll")
        file(GLOB ONNXRUNTIME_LIBS "${ONNXRUNTIME_PATH}/lib/*.lib")
        list(APPEND CAIWEI_RUNTIME_DLLS ${ONNXRUNTIME_DLLS})
        list(APPEND CAIWEI_RUNTIME_INCS "${CMAKE_SOURCE_DIR}/${ONNXRUNTIME_PATH}/include")
    else()
    # -
    endif()
    list(APPEND CAIWEI_RUNTIME_LIBS ${ONNXRUNTIME_LIBS})
endif()

if (ENABLE_CAIWEI_SDL2)
    message(STATUS "Enable SDL2 Player")
    if (ANDROID)
    # -
    elseif(CMAKE_HOST_UNIX)
    # -
    elseif(CMAKE_HOST_WIN32)
        set(SDL2_PATH "deps/sdl2" CACHE STRING "SDL2 Path")
        set(
            CMAKE_PREFIX_PATH
            ${CMAKE_PREFIX_PATH}
            ${SDL2_PATH}
        )
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            file(GLOB SDL2_DLLS "${SDL2_PATH}/debug/bin/*.dll")
        else()
            file(GLOB SDL2_DLLS "${SDL2_PATH}/bin/*.dll")
        endif()
        list(APPEND CAIWEI_MEDIA_DLLS ${SDL2_DLLS})
    else()
    # -
    endif()
    find_package(SDL2 CONFIG REQUIRED)
    list(APPEND CAIWEI_MEDIA_LIBS $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>)
    list(APPEND CAIWEI_MEDIA_LIBS $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>)
endif()

if (ENABLE_CAIWEI_FFMPEG)
    message(STATUS "Enable FFmpeg")
    if(ANDROID)
    # -
    elseif(CMAKE_HOST_UNIX)
        list(APPEND CAIWEI_MEDIA_LIBS avutil avcodec avdevice avformat swscale swresample)
    elseif(CMAKE_HOST_WIN32)
        set(FFMPEG_PATH "deps/ffmpeg" CACHE STRING "FFmpeg Path")
        set(
            CMAKE_PREFIX_PATH
            ${CMAKE_PREFIX_PATH}
            ${FFMPEG_PATH}
        )
        find_package(FFMPEG REQUIRED)
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            file(GLOB FFMPEG_DLLS "${FFMPEG_PATH}/debug/bin/*.dll")
        else()
            file(GLOB FFMPEG_DLLS "${FFMPEG_PATH}/bin/*.dll")
        endif()
        list(APPEND CAIWEI_MEDIA_DLLS ${FFMPEG_DLLS})
        list(APPEND CAIWEI_MEDIA_INCS ${FFMPEG_INCLUDE_DIRS})
        list(APPEND CAIWEI_MEDIA_LIBS ${FFMPEG_LIBRARIES})
    else()
    # -
    endif()
endif()

if (ENABLE_CAIWEI_CPPHTTPLIB)
    message(STATUS "Enable cpp-httplib")
    add_definitions(-DENABLE_CAIWEI_REST)
    if(ANDROID)
    # -
    elseif(CMAKE_HOST_UNIX)
    # -
    elseif(CMAKE_HOST_WIN32)
        set(CPP_HTTPLIB_PATH "deps/cpp-httplib" CACHE STRING "cpp-httplib Path")
        set(
            CMAKE_PREFIX_PATH
            ${CMAKE_PREFIX_PATH}
            ${CPP_HTTPLIB_PATH}
        )
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            file(GLOB CPP_HTTPLIB_DLLS "${CPP_HTTPLIB_PATH}/debug/bin/*.dll")
        else()
            file(GLOB CPP_HTTPLIB_DLLS "${CPP_HTTPLIB_PATH}/bin/*.dll")
        endif()
        find_package(httplib REQUIRED)
        list(APPEND CAIWEI_SERVER_DLLS ${CPP_HTTPLIB_DLLS})
        list(APPEND CAIWEI_SERVER_LIBS $<IF:$<TARGET_EXISTS:cpp-httplib>,cpp-httplib,httplib::httplib>)
    else()
    # -
    endif()
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug/bin)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release/bin)

add_subdirectory(media)
add_subdirectory(runtime)
add_subdirectory(server)
if (ENABLE_CAIWEI_TEST)
    message(STATUS "Build Test")
    add_subdirectory(test)
endif()
