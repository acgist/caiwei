project(caiwei-core VERSION 1.0.0 LANGUAGES C CXX)

set(
    CAIWEI_CORE_SRCS
    third_party/base64/base64.cpp
)
list(APPEND CAIWEI_CORE_INCS third_party)

add_library(
    ${PROJECT_NAME}
    src/env.cpp
    ${CAIWEI_CORE_SRCS}
)
target_link_libraries(
    ${PROJECT_NAME}
    ${CAIWEI_CORE_LIBS}
)
target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
    inc
    ${CAIWEI_CORE_INCS}
)

if (ANDROID)
# -
elseif (CMAKE_HOST_LINUX)
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