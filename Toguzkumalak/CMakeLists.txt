cmake_minimum_required(VERSION 3.10)

# Название проекта
project(Toguzkumalak)

# Устанавливаем стандарт C++
set(CMAKE_CXX_STANDARD 20)

# Указываем каталоги с исходниками
set(SOURCES
    game.cpp
    mcts_selfplay.cpp
    node.cpp
    pch.cpp
    Toguzkumalak.cpp
)

set(HEADERS
    game.h
    mcts_selfplay.h
    node.h
    pch.h
)

# Устанавливаем пути для зависимостей (например, libtorch)
set(LIBTORCH_DIR "C:/Users/Akzhol/source/repos/Toguzkumalak/libtorch")

# Подключаем директории для заголовочных файлов
include_directories(${LIBTORCH_DIR}/include)
include_directories(${LIBTORCH_DIR}/include/torch/csrc/api/include)

# Указываем директорию с библиотеками для линковки
link_directories(${LIBTORCH_DIR}/lib)

# Добавляем исполнимый файл
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Устанавливаем флаги компилятора для конфигураций Debug и Release
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:/DDEBUG /D_CONSOLE /std:c++20>
    $<$<CONFIG:Release>:/DNDEBUG /D_CONSOLE /std:c++20 /O2>
)

# Линковка с необходимыми библиотеками
target_link_libraries(${PROJECT_NAME}
    c10.lib
    torch.lib
    torch_cpu.lib
)

# Настроим копирование DLL файлов после сборки
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${LIBTORCH_DIR}/lib/*.dll"
    $<TARGET_FILE_DIR:${PROJECT_NAME}>
)

# Настройки для платформы x64 и Win32
set_target_properties(${PROJECT_NAME} PROPERTIES
    COMPILE_PDB_NAME "${PROJECT_NAME}.pdb"
    LINK_FLAGS "/SUBSYSTEM:CONSOLE"
)

# Установки для отладки
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Z7")
endif()
