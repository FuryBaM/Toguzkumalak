from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
import sys

# Определяем пути
base_dir = os.path.abspath(os.path.dirname(__file__))  # Директория cython
project_dir = os.path.join(base_dir, "..", "Toguzkumalak")  # Папка с C++ кодом

# Указание всех необходимых файлов
source_files = [
    os.path.join(project_dir, "game.cpp"),
    os.path.join(project_dir, "node.cpp"),
    "node_cython.pyx"
]

# Указание директорий для заголовочных файлов C++
include_dirs = [
    np.get_include(),  # Путь к Numpy
    os.path.dirname(os.path.abspath(__file__)),  # Текущая директория
    project_dir  # Папка с заголовочными файлами
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

# Определение параметров компиляции для Windows и Linux
if sys.platform == "win32":
    extra_compile_args = ["/std:c++20", "/O2", "/fp:fast"]
    extra_link_args = ["/std:c++20"]
else:
    extra_compile_args = ["-std=c++20", "-O3", "-ffast-math", "-march=native"]
    extra_link_args = []

# Определение расширения
extensions = [
    Extension(
        name="mcts",  # Название расширения
        sources=source_files,  # Исходники
        include_dirs=include_dirs,  # Пути к заголовочным файлам
        define_macros=define_macros,  # Определенные макросы
        extra_compile_args=extra_compile_args,  # Флаги компиляции
        extra_link_args=extra_link_args,  # Флаги линковки
        language='c++'  # Используем C++
    )
]

# Запуск установки
setup(
    name="mcts",
    ext_modules=cythonize(
        extensions,
        compiler_directives=compiler_directives,
        annotate=True,  # Генерация аннотированного HTML
    )
)
