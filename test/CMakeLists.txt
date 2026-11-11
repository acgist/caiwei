if(CMAKE_HOST_UNIX)
    find_package(SDL2 QUIET)
elseif(CMAKE_HOST_WIN32)
    set(SDL2_PATH "${CMAKE_SOURCE_DIR}/deps/sdl2" CACHE STRING "SDL2 Path")
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${SDL2_PATH}")
    file(GLOB SDL2_DLLS_DEBUG "${SDL2_PATH}/debug/bin/*.dll")
    file(GLOB SDL2_DLLS_RELEASE "${SDL2_PATH}/bin/*.dll")
    list(APPEND CAIWEI_TEST_DLLS_DEBUG ${SDL2_DLLS_DEBUG})
    list(APPEND CAIWEI_TEST_DLLS_RELEASE ${SDL2_DLLS_RELEASE})
else()
    message(FATAL_ERROR "SDL2不支持的平台")
endif()
find_package(SDL2 REQUIRED)
list(APPEND CAIWEI_TEST_LIBS $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>)
list(APPEND CAIWEI_TEST_LIBS $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>)

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
    caiwei-runtime
    ${CAIWEI_TEST_LIBS}
)

set(
    CAIWEI_TEST_LIST
    test_media_audio
    test_media_image
    test_media_format
    test_media_demuxer
    test_qwen_asr
    test_qwen_llm
    test_qwen_vlm
    test_qwen_embedding
    test_qwen_reranking
    test_yolo_cls
    test_yolo_det
    test_runtime_token
    test_runtime_manager
    test_runtime_request
    test_runtime_runtime
    test_runtime_session
)

foreach(TEST_NAME IN LISTS CAIWEI_TEST_LIST)
    add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    set_target_properties(${TEST_NAME} PROPERTIES FOLDER "test")
    target_link_libraries(${TEST_NAME} caiwei-test)
endforeach()

message(STATUS "CAIWEI_TEST_INCS: ${CAIWEI_TEST_INCS}")
message(STATUS "CAIWEI_TEST_LIBS: ${CAIWEI_TEST_LIBS}")

file(
    GLOB CAIWEI_TEST_RESOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.jpg"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.mp3"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.mp4"
)

if (is_multi_config)
    add_custom_command(
        TARGET caiwei-test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CAIWEI_TEST_RESOURCES} "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
    )
else()
    add_custom_command(
        TARGET caiwei-test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CAIWEI_TEST_RESOURCES} "${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/bin/"
    )
endif()

if(CMAKE_HOST_UNIX)
    # -
elseif(CMAKE_HOST_WIN32)
    message(STATUS "CAIWEI_TEST_DLLS_DEBUG: ${CAIWEI_TEST_DLLS_DEBUG}")
    message(STATUS "CAIWEI_TEST_DLLS_RELEASE: ${CAIWEI_TEST_DLLS_RELEASE}")
    if(CAIWEI_TEST_DLLS_DEBUG AND CAIWEI_TEST_DLLS_RELEASE)
        add_custom_command(
            TARGET caiwei-test POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<IF:$<CONFIG:Debug>,${CAIWEI_TEST_DLLS_DEBUG},${CAIWEI_TEST_DLLS_RELEASE}>" "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        )
    endif()
else()
    # -
endif()
