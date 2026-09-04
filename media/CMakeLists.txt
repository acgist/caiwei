project(caiwei-media VERSION 1.0.0 LANGUAGES C CXX)

set(
    CAIWEI_MEDIA_SRCS
    src/env.cpp
    src/log.cpp
    src/media/text_data.cpp
    src/media/text_tool.cpp
    src/media/audio_tool.cpp
    src/media/audio_data.cpp
    src/media/image_tool.cpp
    src/media/image_data.cpp
    src/media/media.cpp
    src/media/media_data.cpp
    third_party/base64/base64.cpp
)
list(
    APPEND CAIWEI_MEDIA_INCS
    inc
    third_party
)

if(ENABLE_CAIWEI_FFMPEG)
    list(
        APPEND CAIWEI_MEDIA_SRCS
        src/ffmpeg/media.cpp
        src/ffmpeg/media_muxer.cpp
        src/ffmpeg/media_format.cpp
        src/ffmpeg/media_demuxer.cpp
    )
endif()

message(STATUS "CAIWEI_MEDIA_DLLS: ${CAIWEI_MEDIA_DLLS}")
message(STATUS "CAIWEI_MEDIA_INCS: ${CAIWEI_MEDIA_INCS}")
message(STATUS "CAIWEI_MEDIA_LIBS: ${CAIWEI_MEDIA_LIBS}")

add_library(
    ${PROJECT_NAME}
    STATIC
    ${CAIWEI_MEDIA_SRCS}
)
target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC
    ${CAIWEI_MEDIA_LIBS}
)
target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
    ${CAIWEI_MEDIA_INCS}
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
