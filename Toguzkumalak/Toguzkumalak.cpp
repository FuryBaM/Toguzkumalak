#include "pch.h"

#include "mcts_selfplay.h"

int main(int argc, char** argv) {
    int num_games = 25;  // Количество игр по умолчанию
    int cpus = 1;  // Число ядер процессора по умолчанию

    // Обработка аргументов командной строки
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--games" && i + 1 < argc) {
            num_games = std::stoi(argv[++i]);
        }
        else if (arg == "--cpus" && i + 1 < argc) {
            cpus = std::stoi(argv[++i]);
        }
    }

    std::cout << "Number of games: " << num_games << std::endl;
    std::cout << "Number of CPUs: " << cpus << std::endl;

    // Загружаем модель
    std::string model_path = "./model_data/current_trained_net2.pt";  // Пример пути к модели

    // Запуск параллельных процессов для игры
    std::vector<std::thread> threads;
    for (int i = 0; i < cpus; ++i) {
        threads.push_back(std::thread(MCTS_self_play, model_path, num_games, i));
    }

    // Ожидаем завершения всех потоков
    for (auto& t : threads) {
        t.join();
    }

    std::cout << "Self-play is over." << std::endl;

    return 0;
}