from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
import sys

# Указание всех необходимых файлов
source_files = [
    "game.cpp", "node.cpp", "node_cython.pyx"
]

# Указание директорий для включения заголовочных файлов C++
include_dirs = [
    np.get_include(),  # Путь к Numpy
    os.getcwd()  # Текущая директория для заголовочных файлов
]

# Определяем дополнительные макросы
define_macros = [('NPY_NO_DEPRECATED_API', '1')]

# Директивы компилятора для Cython
compiler_directives = {
    "language_level": "3",  # Для Python 3
    "boundscheck": False,   # Отключаем проверку границ
    "initializedcheck": False,  # Отключаем проверку инициализации
    "cdivision": True,      # Разрешаем целочисленное деление
}

# Проверка, используется ли MSVC (cl.exe)
if sys.platform == "win32":
    extra_compile_args = ["/std:c++20", "/O2", "/fp:fast"]
else:
    extra_compile_args = ["-std=c++20", "-O3", "-ffast-math", "-march=native"]

# Определение расширения с правильными флагами компилятора для MSVC
extensions = [
    Extension(
        name="mcts",  # Название вашего расширения
        sources=source_files,  # Указываем pxd файлы
        include_dirs=include_dirs,  # Включаем библиотеки Numpy
        extra_compile_args=extra_compile_args,  # Для MSVC, поддержка C++20
        extra_link_args=['/std:c++20'],     # Линковка с этим флагом
        language='c++'  # Указываем, что код на C++
    )
]

# Использование setuptools вместо distutils для более гибкой сборки
setup(
    name="mcts",
    ext_modules=cythonize(
        extensions,
        compiler_directives=compiler_directives,
        annotate=True,  # Для генерации аннотированных HTML файлов
    )
)
