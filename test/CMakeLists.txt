enable_testing()

option(ENABLE_CAIWEI_SDL2 "SDL2 Player" ON) # SDL2

if(ENABLE_CAIWEI_SDL2)
    message(STATUS "Enable SDL2 Player")
    if(ANDROID)
    # -
    elseif(CMAKE_HOST_UNIX)
    # -
    elseif(CMAKE_HOST_WIN32)
        set(SDL2_PATH "${CMAKE_SOURCE_DIR}/deps/sdl2" CACHE STRING "SDL2 Path")
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
        list(APPEND CAIWEI_TEST_DLLS ${SDL2_DLLS})
    else()
    # -
    endif()
    find_package(SDL2 REQUIRED)
    list(APPEND CAIWEI_TEST_LIBS $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>)
    list(APPEND CAIWEI_TEST_LIBS $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>)
endif()

set(
    CAIWEI_TEST_SRCS
    player.cpp
)

add_library(
    caiwei-test
    STATIC
    ${CAIWEI_TEST_SRCS}
)
set_target_properties(caiwei-test PROPERTIES FOLDER "test")
target_link_libraries(
    caiwei-test
    PUBLIC
    caiwei-media
    caiwei-runtime
    ${CAIWEI_TEST_LIBS}
)

add_executable(
    test_media_env
    test_media_env.cpp
)
add_test(NAME test_media_env COMMAND test_media_env)
set_target_properties(test_media_env PROPERTIES FOLDER "test")
target_link_libraries(
    test_media_env
    caiwei-test
)

add_executable(
    test_media_text
    test_media_text.cpp
)
add_test(NAME test_media_text COMMAND test_media_text)
set_target_properties(test_media_text PROPERTIES FOLDER "test")
target_link_libraries(
    test_media_text
    caiwei-test
)

add_executable(
    test_media_image
    test_media_image.cpp
)
add_test(NAME test_media_image COMMAND test_media_image)
set_target_properties(test_media_image PROPERTIES FOLDER "test")
target_link_libraries(
    test_media_image
    caiwei-test
)

add_executable(
    test_media_format
    test_media_format.cpp
)
add_test(NAME test_media_format COMMAND test_media_format)
set_target_properties(test_media_format PROPERTIES FOLDER "test")
target_link_libraries(
    test_media_format
    caiwei-test
)

add_executable(
    test_media_demuxer
    test_media_demuxer.cpp
)
add_test(NAME test_media_demuxer COMMAND test_media_demuxer)
set_target_properties(test_media_demuxer PROPERTIES FOLDER "test")
target_link_libraries(
    test_media_demuxer
    caiwei-test
)

add_executable(
    test_runtime_asr
    test_runtime_asr.cpp
)
add_test(NAME test_runtime_asr COMMAND test_runtime_asr)
set_target_properties(test_runtime_asr PROPERTIES FOLDER "test")
target_link_libraries(
    test_runtime_asr
    caiwei-test
)

add_executable(
    test_runtime_llm
    test_runtime_llm.cpp
)
add_test(NAME test_runtime_llm COMMAND test_runtime_llm)
set_target_properties(test_runtime_llm PROPERTIES FOLDER "test")
target_link_libraries(
    test_runtime_llm
    caiwei-test
)

add_executable(
    test_runtime_vlm
    test_runtime_vlm.cpp
)
add_test(NAME test_runtime_vlm COMMAND test_runtime_vlm)
set_target_properties(test_runtime_vlm PROPERTIES FOLDER "test")
target_link_libraries(
    test_runtime_vlm
    caiwei-test
)

add_executable(
    test_runtime_embedding
    test_runtime_embedding.cpp
)
add_test(NAME test_runtime_embedding COMMAND test_runtime_embedding)
set_target_properties(test_runtime_embedding PROPERTIES FOLDER "test")
target_link_libraries(
    test_runtime_embedding
    caiwei-test
)

add_executable(
    test_runtime_reranking
    test_runtime_reranking.cpp
)
add_test(NAME test_runtime_reranking COMMAND test_runtime_reranking)
set_target_properties(test_runtime_reranking PROPERTIES FOLDER "test")
target_link_libraries(
    test_runtime_reranking
    caiwei-test
)

add_executable(
    test_runtime_yolo_cls
    test_runtime_yolo_cls.cpp
)
add_test(NAME test_runtime_yolo_cls COMMAND test_runtime_yolo_cls)
set_target_properties(test_runtime_yolo_cls PROPERTIES FOLDER "test")
target_link_libraries(
    test_runtime_yolo_cls
    caiwei-test
)

add_executable(
    test_runtime_yolo_det
    test_runtime_yolo_det.cpp
)
add_test(NAME test_runtime_yolo_det COMMAND test_runtime_yolo_det)
set_target_properties(test_runtime_yolo_det PROPERTIES FOLDER "test")
target_link_libraries(
    test_runtime_yolo_det
    caiwei-test
)

add_executable(
    test_runtime_yolo_seg
    test_runtime_yolo_seg.cpp
)
add_test(NAME test_runtime_yolo_seg COMMAND test_runtime_yolo_seg)
set_target_properties(test_runtime_yolo_seg PROPERTIES FOLDER "test")
target_link_libraries(
    test_runtime_yolo_seg
    caiwei-test
)

add_executable(
    test_runtime_yolo_pose
    test_runtime_yolo_pose.cpp
)
add_test(NAME test_runtime_yolo_pose COMMAND test_runtime_yolo_pose)
set_target_properties(test_runtime_yolo_pose PROPERTIES FOLDER "test")
target_link_libraries(
    test_runtime_yolo_pose
    caiwei-test
)

add_executable(
    test_runtime_rnnoise
    test_runtime_rnnoise.cpp
)
add_test(NAME test_runtime_rnnoise COMMAND test_runtime_rnnoise)
set_target_properties(test_runtime_rnnoise PROPERTIES FOLDER "test")
target_link_libraries(
    test_runtime_rnnoise
    caiwei-test
)

add_executable(
    test_runtime_silero_vad
    test_runtime_silero_vad.cpp
)
add_test(NAME test_runtime_silero_vad COMMAND test_runtime_silero_vad)
set_target_properties(test_runtime_silero_vad PROPERTIES FOLDER "test")
target_link_libraries(
    test_runtime_silero_vad
    caiwei-test
)

add_executable(
    test_runtime_manager
    test_runtime_manager.cpp
)
add_test(NAME test_runtime_manager COMMAND test_runtime_manager)
set_target_properties(test_runtime_manager PROPERTIES FOLDER "test")
target_link_libraries(
    test_runtime_manager
    caiwei-test
)

add_executable(
    test_runtime_runtime
    test_runtime_runtime.cpp
)
add_test(NAME test_runtime_runtime COMMAND test_runtime_runtime)
set_target_properties(test_runtime_runtime PROPERTIES FOLDER "test")
target_link_libraries(
    test_runtime_runtime
    caiwei-test
)

add_executable(
    test_runtime_tokenizer
    test_runtime_tokenizer.cpp
)
add_test(NAME test_runtime_tokenizer COMMAND test_runtime_tokenizer)
set_target_properties(test_runtime_tokenizer PROPERTIES FOLDER "test")
target_link_libraries(
    test_runtime_tokenizer
    caiwei-test
)

message(STATUS "CAIWEI_TEST_DLLS: ${CAIWEI_TEST_DLLS}")
message(STATUS "CAIWEI_TEST_INCS: ${CAIWEI_TEST_INCS}")
message(STATUS "CAIWEI_TEST_LIBS: ${CAIWEI_TEST_LIBS}")

file(COPY ./ DESTINATION ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/bin FILES_MATCHING PATTERN "*.jpg" PATTERN "*.mp3" PATTERN "*.mp4")

if(ANDROID)
# -
elseif(CMAKE_HOST_UNIX)
# -
elseif(CMAKE_HOST_WIN32)
    if(CAIWEI_TEST_DLLS)
        add_custom_command(
            TARGET caiwei-test POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CAIWEI_TEST_DLLS} ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/bin/
        )
    endif()
else()
# -
endif()

