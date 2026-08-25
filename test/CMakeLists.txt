enable_testing()

add_executable(
    test_env
    test_env.cpp
)
add_test(NAME test_env COMMAND test_env)
set_target_properties(test_env PROPERTIES FOLDER "test")
target_link_libraries(
    test_env
    caiwei-core
)

add_executable(
    test_media_format
    test_media_format.cpp
)
add_test(NAME test_media_format COMMAND test_media_format)
set_target_properties(test_media_format PROPERTIES FOLDER "test")
target_link_libraries(
    test_media_format
    caiwei-core
)

add_executable(
    test_media_demuxer
    test_media_demuxer.cpp
)
add_test(NAME test_media_demuxer COMMAND test_media_demuxer)
set_target_properties(test_media_demuxer PROPERTIES FOLDER "test")
target_link_libraries(
    test_media_demuxer
    caiwei-core
)
