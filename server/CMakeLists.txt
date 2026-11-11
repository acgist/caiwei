project(caiwei-server VERSION 1.0.0)

set(
    CAIWEI_SERVER_SRCS
    src/main.cpp
    src/rest/api.cpp
    src/rest/rest.cpp
)

list(
    APPEND CAIWEI_SERVER_INCS
    "${CMAKE_CURRENT_SOURCE_DIR}/inc"
)

list(
    APPEND CAIWEI_SERVER_LIBS
    caiwei-runtime
)

if(CMAKE_HOST_UNIX)
    # -
elseif(CMAKE_HOST_WIN32)
    list(
        APPEND CAIWEI_SERVER_SRCS
        res/win.rc
    )
else()
    # -
endif()

message(STATUS "CAIWEI_SERVER_INCS: ${CAIWEI_SERVER_INCS}")
message(STATUS "CAIWEI_SERVER_LIBS: ${CAIWEI_SERVER_LIBS}")

add_executable(
    ${PROJECT_NAME}
    ${CAIWEI_SERVER_SRCS}
)
target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
    ${CAIWEI_SERVER_LIBS}
)
target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
    ${CAIWEI_SERVER_INCS}
)

file(
    GLOB CAIWEI_RESOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/res/*.ico"
    "${CMAKE_CURRENT_SOURCE_DIR}/res/*.txt"
)

add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CAIWEI_RESOURCES} "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
)

if(CMAKE_HOST_UNIX)
    install(DIRECTORY res/ DESTINATION bin FILES_MATCHING PATTERN "*.ico" PATTERN "*.txt")
    install(
        TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        PUBLIC_HEADER DESTINATION include
    )
elseif(CMAKE_HOST_WIN32)
    # /SUBSYSTEM:CONSOLE
    # /SUBSYSTEM:WINDOWS
    # /entry:mainCRTStartup
    # /entry:WinMainCRTStartup
    # C = C | CP = C++
    # /MD  MSVCRT.LIB   多线程DLL的Release版本
    # /MDd MSVCRTD.LIB  多线程DLL的Debug版本
    # /MD  MSVCPRT.LIB  多线程DLL的Release版本
    # /MDd MSVCPRTD.LIB 多线程DLL的Debug版本
    # /MT  LIBCMT.LIB   多线程静态链接的Release版本
    # /MTd LIBCMTD.LIB  多线程静态链接的Debug版本
    # /MT  LIBCPMT.LIB  多线程静态链接的Release版本
    # /MTd LIBCPMTD.LIB 多线程静态链接的Debug版本
    # set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    # set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL$<$<CONFIG:Debug>:Debug>")
    # set_property(TARGET project PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    # set_property(TARGET project PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreadedDLL$<$<CONFIG:Debug>:Debug>")
    target_link_options(
        ${PROJECT_NAME} PRIVATE
        $<$<CONFIG:Debug>:/SUBSYSTEM:CONSOLE>
        $<$<CONFIG:Release>:/SUBSYSTEM:WINDOWS /entry:mainCRTStartup>
    )
    message(STATUS "CAIWEI_SERVER_DLLS: ${CAIWEI_SERVER_DLLS}")
    message(STATUS "CAIWEI_SERVER_DLLS_DEBUG: ${CAIWEI_SERVER_DLLS_DEBUG}")
    message(STATUS "CAIWEI_SERVER_DLLS_RELEASE: ${CAIWEI_SERVER_DLLS_RELEASE}")
    if(CAIWEI_SERVER_DLLS)
        add_custom_command(
            TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CAIWEI_SERVER_DLLS} "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
        )
    endif()
    if(CAIWEI_SERVER_DLLS_DEBUG AND CAIWEI_SERVER_DLLS_RELEASE)
        add_custom_command(
            TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<IF:$<CONFIG:Debug>,${CAIWEI_SERVER_DLLS_DEBUG},${CAIWEI_SERVER_DLLS_RELEASE}>" "${CMAKE_BINARY_DIR}/$<CONFIG>/bin/"
            COMMAND_EXPAND_LISTS
            VERBATIM
        )
    endif()
else()
    # -
endif()
