# Toguzkumalak

Реализация игры и обучение нейросети для тогузкумалака. Код написан на C++ и Python, модель использует библиотеку LibTorch.

## Требования
- компилятор C++17 и CMake 3.10+
- Python 3.8 или новее
- LibTorch 2.6.0
  - CPU: `libtorch-cxx11-abi-shared-with-deps-2.6.0+cpu`
  - CUDA 12.4: `libtorch-cxx11-abi-shared-with-deps-2.6.0+cu124`
- программа тестировалась на PyTorch/LibTorch 2.6.0 (CPU и CUDA 12.4)
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
2. Отредактируйте секцию `[train]` в `Toguzkumalak/config.ini` или передайте параметры через командную строку.
3. Запустите обучение:
```bash
./Toguzkumalak --mode train --dataset ./datasets/combined/combined_dataset.bin
```
Для Windows используйте `Toguzkumalak.exe`. Дополнительные параметры: `--cpus`, `--epochs`, `--model` и т.д.

## Использование конфигурационных файлов
Все параметры можно указать в отдельном файле и передать его через `--config путь/к/файлу`.
По умолчанию используется `config.ini`. Точка входа программы находится в `Toguzkumalak.cpp`.

### Пример `config.ini`
```ini
[global]
mode = train
cpus = 2

[train]
model = ./model_data/best.pt
save = ./model_data/best.pt
dataset = ./datasets/combined/combined_dataset.bin
epochs = 100
lr = 1e-4
lr_step = 10
gamma = 0.2
batch_size = 32

Запуск обучения с конфигурацией:

```bash
./Toguzkumalak --config Toguzkumalak/config.ini

### Генерация датасета
Чтобы собрать набор данных, запустите режим selfplay:

```bash
./Toguzkumalak --config Toguzkumalak/config.ini
```
Файлы будут сохранены в каталог, указанный в параметре `save`.

### Режим arena
Автоматизирует цикл "selfplay → обучение". Параметры задаются в секции `[arena]` конфигурации.

```bash
./Toguzkumalak --mode arena --config Toguzkumalak/config.ini
```

### Загрузка и сохранение весов в C++
Работать с отдельным файлом весов можно через методы `save_weights` и `load_weights` класса `TNET`:

```cpp
std::shared_ptr<TNET> model = std::make_shared<TNET>();
model->load_weights("./model_data/weights.dat");
// ... обучение или инференс ...
model->save_weights("./model_data/weights.dat");
```
