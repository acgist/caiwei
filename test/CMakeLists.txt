enable_testing()

add_executable(
    test_env
    test_env.cpp
)
add_test(NAME test_env COMMAND test_env)
set_target_properties(test_env PROPERTIES FOLDER "test-media")
target_link_libraries(
    test_env
    caiwei-media
)

add_executable(
    test_media_format
    test_media_format.cpp
)
add_test(NAME test_media_format COMMAND test_media_format)
set_target_properties(test_media_format PROPERTIES FOLDER "test-media")
target_link_libraries(
    test_media_format
    caiwei-media
)

add_executable(
    test_media_demuxer
    test_media_demuxer.cpp
)
add_test(NAME test_media_demuxer COMMAND test_media_demuxer)
set_target_properties(test_media_demuxer PROPERTIES FOLDER "test-media")
target_link_libraries(
    test_media_demuxer
    caiwei-media
)

add_executable(
    test_model_asr
    test_model_asr.cpp
)
add_test(NAME test_model_asr COMMAND test_model_asr)
set_target_properties(test_model_asr PROPERTIES FOLDER "test-runtime")
target_link_libraries(
    test_model_asr
    caiwei-runtime
)

add_executable(
    test_model_tts
    test_model_tts.cpp
)
add_test(NAME test_model_tts COMMAND test_model_tts)
set_target_properties(test_model_tts PROPERTIES FOLDER "test-runtime")
target_link_libraries(
    test_model_tts
    caiwei-runtime
)

add_executable(
    test_model_llm
    test_model_llm.cpp
)
add_test(NAME test_model_llm COMMAND test_model_llm)
set_target_properties(test_model_llm PROPERTIES FOLDER "test-runtime")
target_link_libraries(
    test_model_llm
    caiwei-runtime
)

add_executable(
    test_model_vlm
    test_model_vlm.cpp
)
add_test(NAME test_model_vlm COMMAND test_model_vlm)
set_target_properties(test_model_vlm PROPERTIES FOLDER "test-runtime")
target_link_libraries(
    test_model_vlm
    caiwei-runtime
)

add_executable(
    test_model_mllm
    test_model_mllm.cpp
)
add_test(NAME test_model_mllm COMMAND test_model_mllm)
set_target_properties(test_model_mllm PROPERTIES FOLDER "test-runtime")
target_link_libraries(
    test_model_mllm
    caiwei-runtime
)

add_executable(
    test_model_embedding
    test_model_embedding.cpp
)
add_test(NAME test_model_embedding COMMAND test_model_embedding)
set_target_properties(test_model_embedding PROPERTIES FOLDER "test-runtime")
target_link_libraries(
    test_model_embedding
    caiwei-runtime
)

add_executable(
    test_model_reranking
    test_model_reranking.cpp
)
add_test(NAME test_model_reranking COMMAND test_model_reranking)
set_target_properties(test_model_reranking PROPERTIES FOLDER "test-runtime")
target_link_libraries(
    test_model_reranking
    caiwei-runtime
)

add_executable(
    test_model_yolo_cls
    test_model_yolo_cls.cpp
)
add_test(NAME test_model_yolo_cls COMMAND test_model_yolo_cls)
set_target_properties(test_model_yolo_cls PROPERTIES FOLDER "test-runtime")
target_link_libraries(
    test_model_yolo_cls
    caiwei-runtime
)

add_executable(
    test_model_yolo_det
    test_model_yolo_det.cpp
)
add_test(NAME test_model_yolo_det COMMAND test_model_yolo_det)
set_target_properties(test_model_yolo_det PROPERTIES FOLDER "test-runtime")
target_link_libraries(
    test_model_yolo_det
    caiwei-runtime
)

add_executable(
    test_model_yolo_seg
    test_model_yolo_seg.cpp
)
add_test(NAME test_model_yolo_seg COMMAND test_model_yolo_seg)
set_target_properties(test_model_yolo_seg PROPERTIES FOLDER "test-runtime")
target_link_libraries(
    test_model_yolo_seg
    caiwei-runtime
)

add_executable(
    test_model_yolo_sem
    test_model_yolo_sem.cpp
)
add_test(NAME test_model_yolo_sem COMMAND test_model_yolo_sem)
set_target_properties(test_model_yolo_sem PROPERTIES FOLDER "test-runtime")
target_link_libraries(
    test_model_yolo_sem
    caiwei-runtime
)

add_executable(
    test_model_yolo_obb
    test_model_yolo_obb.cpp
)
add_test(NAME test_model_yolo_obb COMMAND test_model_yolo_obb)
set_target_properties(test_model_yolo_obb PROPERTIES FOLDER "test-runtime")
target_link_libraries(
    test_model_yolo_obb
    caiwei-runtime
)

add_executable(
    test_model_yolo_pose
    test_model_yolo_pose.cpp
)
add_test(NAME test_model_yolo_pose COMMAND test_model_yolo_pose)
set_target_properties(test_model_yolo_pose PROPERTIES FOLDER "test-runtime")
target_link_libraries(
    test_model_yolo_pose
    caiwei-runtime
)

add_executable(
    test_model_yolo_depth
    test_model_yolo_depth.cpp
)
add_test(NAME test_model_yolo_depth COMMAND test_model_yolo_depth)
set_target_properties(test_model_yolo_depth PROPERTIES FOLDER "test-runtime")
target_link_libraries(
    test_model_yolo_depth
    caiwei-runtime
)

add_executable(
    test_model_rnnoise
    test_model_rnnoise.cpp
)
add_test(NAME test_model_rnnoise COMMAND test_model_rnnoise)
set_target_properties(test_model_rnnoise PROPERTIES FOLDER "test-runtime")
target_link_libraries(
    test_model_rnnoise
    caiwei-runtime
)

add_executable(
    test_model_silero_vad
    test_model_silero_vad.cpp
)
add_test(NAME test_model_silero_vad COMMAND test_model_silero_vad)
set_target_properties(test_model_silero_vad PROPERTIES FOLDER "test-runtime")
target_link_libraries(
    test_model_silero_vad
    caiwei-runtime
)

if(ANDROID)
# -
elseif(CMAKE_HOST_UNIX)
# -
elseif(CMAKE_HOST_WIN32)
    set(
        CMAKE_PREFIX_PATH
        ${CMAKE_PREFIX_PATH}
        "${CMAKE_SOURCE_DIR}/deps/opencv"
    )
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        file(GLOB OPENCV_DLLS "${CMAKE_SOURCE_DIR}/deps/opencv/debug/bin/*.dll")
    else()
        file(GLOB OPENCV_DLLS "${CMAKE_SOURCE_DIR}/deps/opencv/bin/*.dll")
    endif()
else()
# -
endif()

find_package(OpenCV)

if (OpenCV_FOUND)
    add_executable(
        test_cv_opencv
        test_cv_opencv.cpp
    )
    add_test(NAME test_cv_opencv COMMAND test_cv_opencv)
    set_target_properties(test_cv_opencv PROPERTIES FOLDER "test-media")
    target_link_libraries(
        test_cv_opencv
        caiwei-media
        ${OpenCV_LIBS}
    )
    add_custom_command(
        TARGET test_cv_opencv POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OPENCV_DLLS} ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/bin/
    )
endif ()
