#include "pch.h"

#include "mcts_selfplay.h"

int main(int argc, char** argv) {
    int num_games = 25;
    int cpus = std::thread::hardware_concurrency();
    std::string save_path = "";
    std::string model_path = "";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--games" && i + 1 < argc) {
            num_games = std::stoi(argv[++i]);
        }
        else if (arg == "--cpus" && i + 1 < argc) {
            cpus = std::stoi(argv[++i]);
        }
        else if (arg == "--save" && i + 1 < argc) {
            save_path = argv[++i];
        }
        else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        }
    }

    if (save_path.empty()) {
        save_path = "./datasets/all/";
    }
    if (model_path.empty()) {
        model_path = "./model_data/best.pt";
    }

    std::cout << "Number of games: " << num_games << std::endl;
    std::cout << "Number of CPUs: " << cpus << std::endl;
    std::cout << "Save path: " << save_path << std::endl;
    std::cout << "Model path: " << model_path << std::endl;

    std::vector<std::thread> threads;
    for (int i = 0; i < cpus; ++i) {
        threads.push_back(std::thread(MCTS_self_play, model_path, save_path, num_games, i));
    }

    for (auto& t : threads) {
        t.join();
    }

    std::cout << "Self-play is over." << std::endl;

    return 0;
}