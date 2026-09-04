enable_testing()

add_executable(
    test_media_env
    test_media_env.cpp
)
add_test(NAME test_media_env COMMAND test_media_env)
set_target_properties(test_media_env PROPERTIES FOLDER "test-media")
target_link_libraries(
    test_media_env
    caiwei-media
)

add_executable(
    test_media_text
    test_media_text.cpp
)
add_test(NAME test_media_text COMMAND test_media_text)
set_target_properties(test_media_text PROPERTIES FOLDER "test-media")
target_link_libraries(
    test_media_text
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
    test_context_asr
    test_context_asr.cpp
)
add_test(NAME test_context_asr COMMAND test_context_asr)
set_target_properties(test_context_asr PROPERTIES FOLDER "test-runtime")
target_link_libraries(
    test_context_asr
    caiwei-runtime
)

add_executable(
    test_context_tts
    test_context_tts.cpp
)
add_test(NAME test_context_tts COMMAND test_context_tts)
set_target_properties(test_context_tts PROPERTIES FOLDER "test-runtime")
target_link_libraries(
    test_context_tts
    caiwei-runtime
)

add_executable(
    test_context_llm
    test_context_llm.cpp
)
add_test(NAME test_context_llm COMMAND test_context_llm)
set_target_properties(test_context_llm PROPERTIES FOLDER "test-runtime")
target_link_libraries(
    test_context_llm
    caiwei-runtime
)

add_executable(
    test_context_vlm
    test_context_vlm.cpp
)
add_test(NAME test_context_vlm COMMAND test_context_vlm)
set_target_properties(test_context_vlm PROPERTIES FOLDER "test-runtime")
target_link_libraries(
    test_context_vlm
    caiwei-runtime
)

add_executable(
    test_context_mllm
    test_context_mllm.cpp
)
add_test(NAME test_context_mllm COMMAND test_context_mllm)
set_target_properties(test_context_mllm PROPERTIES FOLDER "test-runtime")
target_link_libraries(
    test_context_mllm
    caiwei-runtime
)

add_executable(
    test_context_embedding
    test_context_embedding.cpp
)
add_test(NAME test_context_embedding COMMAND test_context_embedding)
set_target_properties(test_context_embedding PROPERTIES FOLDER "test-runtime")
target_link_libraries(
    test_context_embedding
    caiwei-runtime
)

add_executable(
    test_context_reranking
    test_context_reranking.cpp
)
add_test(NAME test_context_reranking COMMAND test_context_reranking)
set_target_properties(test_context_reranking PROPERTIES FOLDER "test-runtime")
target_link_libraries(
    test_context_reranking
    caiwei-runtime
)

add_executable(
    test_context_yolo_cls
    player.cpp
    test_context_yolo_cls.cpp
)
add_test(NAME test_context_yolo_cls COMMAND test_context_yolo_cls)
set_target_properties(test_context_yolo_cls PROPERTIES FOLDER "test-runtime")
target_link_libraries(
    test_context_yolo_cls
    caiwei-runtime
)

add_executable(
    test_context_yolo_det
    player.cpp
    test_context_yolo_det.cpp
)
add_test(NAME test_context_yolo_det COMMAND test_context_yolo_det)
set_target_properties(test_context_yolo_det PROPERTIES FOLDER "test-runtime")
target_link_libraries(
    test_context_yolo_det
    caiwei-runtime
)

add_executable(
    test_context_yolo_seg
    test_context_yolo_seg.cpp
)
add_test(NAME test_context_yolo_seg COMMAND test_context_yolo_seg)
set_target_properties(test_context_yolo_seg PROPERTIES FOLDER "test-runtime")
target_link_libraries(
    test_context_yolo_seg
    caiwei-runtime
)

add_executable(
    test_context_yolo_sem
    test_context_yolo_sem.cpp
)
add_test(NAME test_context_yolo_sem COMMAND test_context_yolo_sem)
set_target_properties(test_context_yolo_sem PROPERTIES FOLDER "test-runtime")
target_link_libraries(
    test_context_yolo_sem
    caiwei-runtime
)

add_executable(
    test_context_yolo_obb
    test_context_yolo_obb.cpp
)
add_test(NAME test_context_yolo_obb COMMAND test_context_yolo_obb)
set_target_properties(test_context_yolo_obb PROPERTIES FOLDER "test-runtime")
target_link_libraries(
    test_context_yolo_obb
    caiwei-runtime
)

add_executable(
    test_context_yolo_pose
    test_context_yolo_pose.cpp
)
add_test(NAME test_context_yolo_pose COMMAND test_context_yolo_pose)
set_target_properties(test_context_yolo_pose PROPERTIES FOLDER "test-runtime")
target_link_libraries(
    test_context_yolo_pose
    caiwei-runtime
)

add_executable(
    test_context_yolo_depth
    test_context_yolo_depth.cpp
)
add_test(NAME test_context_yolo_depth COMMAND test_context_yolo_depth)
set_target_properties(test_context_yolo_depth PROPERTIES FOLDER "test-runtime")
target_link_libraries(
    test_context_yolo_depth
    caiwei-runtime
)

add_executable(
    test_context_rnnoise
    test_context_rnnoise.cpp
)
add_test(NAME test_context_rnnoise COMMAND test_context_rnnoise)
set_target_properties(test_context_rnnoise PROPERTIES FOLDER "test-runtime")
target_link_libraries(
    test_context_rnnoise
    caiwei-runtime
)

add_executable(
    test_context_silero_vad
    test_context_silero_vad.cpp
)
add_test(NAME test_context_silero_vad COMMAND test_context_silero_vad)
set_target_properties(test_context_silero_vad PROPERTIES FOLDER "test-runtime")
target_link_libraries(
    test_context_silero_vad
    caiwei-runtime
)

if(OpenCV_FOUND)
    add_executable(
        test_cv_opencv
        test_cv_opencv.cpp
    )
    add_test(NAME test_cv_opencv COMMAND test_cv_opencv)
    set_target_properties(test_cv_opencv PROPERTIES FOLDER "test-media")
    target_link_libraries(
        test_cv_opencv
        caiwei-media
    )
endif()

file(COPY ./ DESTINATION ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/bin FILES_MATCHING PATTERN "*.jpg" PATTERN "*.mp4")
