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

# Заглушка для `nvToolsExt`, чтобы избежать ошибки


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
    train.cpp
    mcts_selfplay.cpp
    node.cpp
    pch.cpp
    Toguzkumalak.cpp
    tnet.cpp
)

# Создаём исполняемый файл
add_executable(${PROJECT_NAME} ${SOURCES})

# Подключаем предкомпилированный заголовок (PCH)
target_precompile_headers(${PROJECT_NAME} PRIVATE pch.h)

# Подключаем заголовочные файлы и библиотеки Torch
target_include_directories(${PROJECT_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

# Подключаем OpenMP
find_package(OpenMP REQUIRED COMPONENTS C CXX)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)

    # Для GCC/Clang используем -fopenmp, а для MSVC - /openmp
    if(MSVC)
        target_compile_options(${PROJECT_NAME} PRIVATE /openmp)
    else()
        target_compile_options(${PROJECT_NAME} PRIVATE -fopenmp)
    endif()
endif()

# Опция для включения CUDA
option(USE_CUDA "Enable CUDA support" OFF)

if(USE_CUDA)
    # Настройка CUDA после Torch и OpenMP
    find_package(CUDA REQUIRED)
    if(CUDA_FOUND)
        message(STATUS "CUDA found: ${CUDA_TOOLKIT_ROOT_DIR}")
        
        # Убедитесь, что флаги для компилятора CUDA (nvcc) только для CUDA файлов
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        
        # Флаги для CUDA компилятора
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 --expt-extended-lambda")
        
        target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}" "${CUDA_LIBRARIES}")
        target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})
    else()
        message(WARNING "CUDA not found, falling back to CPU mode.")
    endif()
else()
    message(STATUS "CUDA support is disabled, building for CPU.")
endif()

# Добавляем pthread для Linux
if(UNIX)
    target_link_libraries(${PROJECT_NAME} pthread)
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
