# Toguzkumalak

Реализация игры и обучение нейросети для тогузкумалака. Код написан на C++ и Python, модель использует библиотеку LibTorch.

## Требования
- компилятор C++17 и CMake 3.10+
- Python 3.8 или новее
- LibTorch 2.2
  - CPU: `libtorch-cxx11-abi-shared-with-deps-2.2.0+cpu`
  - CUDA 12.1: `libtorch-cxx11-abi-shared-with-deps-2.2.0+cu121`
- при сборке с GPU понадобится установленный CUDA Toolkit

## Подготовка LibTorch
1. Скачайте соответствующую версию LibTorch с [страницы загрузки PyTorch](https://pytorch.org/get-started/locally/).
2. Распакуйте архив в каталог `libtorch` в корне репозитория.

## Сборка
### Linux
```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release [-DUSE_CUDA=ON]
cmake --build . --config Release
```

### Windows
```bat
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release [-DUSE_CUDA=ON]
cmake --build . --config Release
```
Исполняемый файл появится в каталоге `build`.

## Обучение модели
1. Подготовьте датасет (пример находится в каталоге `datasets`).
2. Отредактируйте `Toguzkumalak/train_config.txt` или передайте параметры через командную строку.
3. Запустите обучение:
```bash
./Toguzkumalak --mode train --dataset ./datasets/combined/combined_dataset.bin
```
Для Windows используйте `Toguzkumalak.exe`. Дополнительные параметры: `--cpus`, `--epochs`, `--model` и т.д.
