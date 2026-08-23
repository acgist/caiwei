enable_testing()

add_executable(
    test_qwen3_asr
    test_qwen3_asr.cpp
)
add_test(NAME test_qwen3_asr COMMAND test_qwen3_asr)
set_target_properties(test_qwen3_asr PROPERTIES FOLDER "test")
target_link_libraries(
    test_qwen3_asr
    caiwei-core
)
