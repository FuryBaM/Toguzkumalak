#include "pch.h"
#include "arena.h"
#include <filesystem>

void combine_datasets(const std::string& folder, const std::string& output_file) {
    GameState combined;
    for (const auto& entry : std::filesystem::directory_iterator(folder)) {
        if (entry.is_regular_file() && entry.path().extension() == ".bin") {
            GameState part;
            part.load(entry.path().string());
            combined.states.insert(combined.states.end(), part.states.begin(), part.states.end());
            combined.policies.insert(combined.policies.end(), part.policies.begin(), part.policies.end());
            combined.values.insert(combined.values.end(), part.values.begin(), part.values.end());
        }
    }
    if (!combined.states.empty()) {
        combined.save(output_file);
    }
}

void run_arena(Config& config) {
    int cpus = config.get<int>("cpus", std::thread::hardware_concurrency(), 0);

    config.setSection("selfplay");
    int games = config.get<int>("games", 10, 0);
    bool use_omp = config.get<bool>("openmp", false, 0);
    int num_reads = config.get<int>("num_reads", 800, 0);
    float temperature = config.get<float>("temperature", 1.0f, 0);
    int temperature_cutoff = config.get<int>("temperature_cutoff", 30, 0);
    std::string sp_model = std::filesystem::absolute(config.get<std::string>("model", "./model_data/best_optimized.pt", 0)).string();
    std::string save_path = std::filesystem::absolute(config.get<std::string>("save", "./arena/selfplay", 0)).string();

    config.setSection("train");
    int epochs = config.get<int>("epochs", 100, 0);
    double lr = config.get<double>("lr", 1e-4, 0);
    int lr_step = config.get<int>("lr_step", 10, 0);
    double gamma = config.get<double>("gamma", 0.2, 0);
    int batch_size = config.get<int>("batch_size", 32, 0);
    std::string dataset_path = std::filesystem::absolute(config.get<std::string>("dataset", "./arena/combined.bin", 0)).string();
    std::string weights_path = std::filesystem::absolute(config.get<std::string>("save_weights", "./model_data/weights.dat", 0)).string();
    std::string model_path = std::filesystem::absolute(config.get<std::string>("model", "./model_data/best.pt", 0)).string();
    std::string save_model = std::filesystem::absolute(config.get<std::string>("save", "./model_data/best.pt", 0)).string();

    std::filesystem::create_directories(save_path);
    std::filesystem::create_directories(std::filesystem::path(dataset_path).parent_path());

    MCTSSelfPlayConfig sp_cfg(games, num_reads, temperature, temperature_cutoff);

    if (cpus > 1) {
        if (use_omp) {
#pragma omp parallel for
            for (int i = 0; i < cpus; ++i) {
                MCTS_self_play(sp_model, save_path, i, true, sp_cfg);
            }
        } else {
            std::vector<std::thread> threads;
            for (int i = 0; i < cpus; ++i) {
                threads.emplace_back(MCTS_self_play, sp_model, save_path, i, true, sp_cfg);
            }
            for (auto& t : threads) {
                if (t.joinable()) t.join();
            }
        }
    } else {
        MCTS_self_play(sp_model, save_path, 0, false, sp_cfg);
    }

    combine_datasets(save_path, dataset_path);

    TrainConfig train_cfg(epochs, lr, lr_step, gamma, batch_size);
    auto model = std::make_shared<TNET>();
    try {
        torch::load(model, model_path);
    } catch (...) {
    }
    start_training(model, dataset_path, 1, train_cfg);
    torch::save(model, save_model);
    model->save_weights(weights_path);
}
