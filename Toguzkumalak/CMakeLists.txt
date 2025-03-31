cmake_minimum_required(VERSION 3.10)
project(Toguzkumalak)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Устанавливаем путь к libtorch (можно переопределить извне)
if(NOT DEFINED LIBTORCH_DIR)
    get_filename_component(LIBTORCH_DIR "${CMAKE_SOURCE_DIR}/../libtorch" REALPATH)
endif()
set(CMAKE_PREFIX_PATH "${LIBTORCH_DIR}")

message(STATUS "LIBTORCH_DIR: ${LIBTORCH_DIR}")
# Подключаем Torch
find_package(Torch REQUIRED)

if (WIN32)
file(GLOB TORCH_DLLS "${LIBTORCH_DIR}/lib/*.dll")
message(STATUS "TORCH_DLLS found: ${TORCH_DLLS}")

if(NOT TORCH_DLLS)
    message(WARNING "No Torch DLLs found! Check if files exist in ${LIBTORCH_DIR}/lib")
endif()
endif()

# Исходники
set(SOURCES
    game.cpp
    mcts_selfplay.cpp
    node.cpp
    pch.cpp
    Toguzkumalak.cpp
)

# Создаём исполняемый файл
add_executable(${PROJECT_NAME} ${SOURCES})

# Подключаем предкомпилированный заголовок (PCH)
target_precompile_headers(${PROJECT_NAME} PRIVATE pch.h)

# Подключаем заголовочные файлы и библиотеки Torch
target_include_directories(${PROJECT_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

# Добавляем pthread для Linux
if(UNIX)
    target_link_libraries(${PROJECT_NAME} pthread)
endif()

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
    target_compile_options(${PROJECT_NAME} PRIVATE -fopenmp)
endif()

# Копируем DLL только в Windows
if(WIN32)
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${TORCH_DLLS} $<TARGET_FILE_DIR:${PROJECT_NAME}>
)
endif()

# Устанавливаем компиляторские флаги
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<CONFIG:Debug>:/DDEBUG /D_CONSOLE /openmp /std:c++20>
        $<$<CONFIG:Release>:/DNDEBUG /D_CONSOLE /openmp /std:c++20 /O2 /W3 /wd4100 /wd4189 /wd4244 /wd4267>
    )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Z7")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<CONFIG:Debug>:-DDEBUG -D_CONSOLE -std=c++20 -g -fopenmp>
        $<$<CONFIG:Release>:-DNDEBUG -D_CONSOLE -std=c++20 -O2 -fopenmp -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -Wno-sign-compare>
    )
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()

# Windows-специфичные свойства
if(WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        COMPILE_PDB_NAME "${PROJECT_NAME}.pdb"
        LINK_FLAGS "/SUBSYSTEM:CONSOLE"
    )
endif()
add_definitions(-D_BUILD_MCTS)
