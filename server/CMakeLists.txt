project(caiwei-server VERSION 1.0.0)

set(
    CAIWEI_SERVER_SRCS
    src/main.cpp
    src/rest/api.cpp
    src/rest/rest.cpp
)

list(APPEND CAIWEI_SERVER_INCS "${CMAKE_CURRENT_SOURCE_DIR}/inc")
list(APPEND CAIWEI_SERVER_LIBS caiwei-runtime)

if(CMAKE_HOST_UNIX)
    # -
elseif(CMAKE_HOST_WIN32)
    list(APPEND CAIWEI_SERVER_SRCS res/win.rc)
else()
    # -
endif()

message(STATUS "CAIWEI_SERVER_INCS: ${CAIWEI_SERVER_INCS}")
message(STATUS "CAIWEI_SERVER_LIBS: ${CAIWEI_SERVER_LIBS}")

add_executable(${PROJECT_NAME} ${CAIWEI_SERVER_SRCS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${CAIWEI_SERVER_LIBS})
target_include_directories(${PROJECT_NAME} PRIVATE ${CAIWEI_SERVER_INCS})

if(CMAKE_HOST_UNIX)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)
elseif(CMAKE_HOST_WIN32)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:C>:/W2> $<$<COMPILE_LANGUAGE:CXX>:/W2>)
else()
    # -
endif()

install(
    TARGETS ${PROJECT_NAME}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES "${CMAKE_SOURCE_DIR}/LICENSE" "${CMAKE_SOURCE_DIR}/README.md" DESTINATION ".")

file(
    GLOB CAIWEI_RESOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/res/*.ico"
    "${CMAKE_CURRENT_SOURCE_DIR}/res/*.txt"
    "${CMAKE_CURRENT_SOURCE_DIR}/res/*.html"
)
add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CAIWEI_RESOURCES} "${CMAKE_BINARY_DIR}/$<IF:$<CONFIG:Debug>,Debug,Release>/bin/"
    COMMENT "server-copy-resources"
)
install(DIRECTORY res/ DESTINATION bin FILES_MATCHING PATTERN "*.ico" PATTERN "*.txt" PATTERN "*.html")

if(CMAKE_HOST_UNIX)
    # -
elseif(CMAKE_HOST_WIN32)
    # C = C | CP = C++
    # /MD  MSVCRT.lib  MSVCPRT.lib  多线程动态CRT Release
    # /MDd MSVCRTD.lib MSVCPRTD.lib 多线程动态CRT Debug
    # /MT  LIBCMT.lib  LIBCPMT.lib  多线程静态CRT Release
    # /MTd LIBCMTD.lib LIBCPMTD.lib 多线程静态CRT Debug
    # set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    # set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL$<$<CONFIG:Debug>:Debug>")
    # set_property(TARGET project PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    # set_property(TARGET project PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreadedDLL$<$<CONFIG:Debug>:Debug>")
    # /SUBSYSTEM:CONSOLE
    # /SUBSYSTEM:WINDOWS
    # /entry:mainCRTStartup
    # /entry:WinMainCRTStartup
    target_link_options(
        ${PROJECT_NAME} PRIVATE
        $<$<CONFIG:Debug>:/SUBSYSTEM:CONSOLE>
        $<$<CONFIG:Release>:/SUBSYSTEM:CONSOLE>
    )
    message(STATUS "CAIWEI_SERVER_DLLS: ${CAIWEI_SERVER_DLLS}")
    message(STATUS "CAIWEI_SERVER_DLLS_DEBUG: ${CAIWEI_SERVER_DLLS_DEBUG}")
    message(STATUS "CAIWEI_SERVER_DLLS_RELEASE: ${CAIWEI_SERVER_DLLS_RELEASE}")
    if(CAIWEI_SERVER_DLLS)
        add_custom_command(
            TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CAIWEI_SERVER_DLLS} "${CMAKE_BINARY_DIR}/$<IF:$<CONFIG:Debug>,Debug,Release>/bin/"
            COMMENT "server-copy-dlls-common"
        )
        install(FILES ${CAIWEI_SERVER_DLLS} DESTINATION bin)
    endif()
    if(CAIWEI_SERVER_DLLS_DEBUG AND CAIWEI_SERVER_DLLS_RELEASE)
        add_custom_command(
            TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<IF:$<CONFIG:Debug>,${CAIWEI_SERVER_DLLS_DEBUG},${CAIWEI_SERVER_DLLS_RELEASE}>" "${CMAKE_BINARY_DIR}/$<IF:$<CONFIG:Debug>,Debug,Release>/bin/"
            COMMENT "server-copy-dlls-custom"
            COMMAND_EXPAND_LISTS
            VERBATIM
        )
        install(FILES "$<IF:$<CONFIG:Debug>,${CAIWEI_SERVER_DLLS_DEBUG},${CAIWEI_SERVER_DLLS_RELEASE}>" DESTINATION bin)
    endif()
else()
    # -
endif()
