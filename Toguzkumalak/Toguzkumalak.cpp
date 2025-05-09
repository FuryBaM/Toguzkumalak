﻿#include "pch.h"
#include "mcts_selfplay.h"
#include "train.h"
#include "tnet.h"

int main(int argc, char** argv) {
    std::string config_path = "config.txt";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        }
    }

    config_path = std::filesystem::absolute(config_path).string();
    Config config(config_path);

    std::string mode = config.get<std::string>("mode", "selfplay", 0);
    int cpus = config.get<int>("cpus", std::thread::hardware_concurrency(), 0);
    torch::set_num_threads(std::thread::hardware_concurrency()); // или столько, сколько ядер
    torch::set_num_interop_threads(std::thread::hardware_concurrency());
    omp_set_num_threads(std::thread::hardware_concurrency());

    std::string model_path = std::filesystem::absolute(
        config.get<std::string>("model", "./model_data/best_optimized.pt", 0)
    ).string();

    std::cout << "Mode: " << mode << std::endl;
    std::cout << "Model path: " << model_path << std::endl;

    if (mode == "selfplay") {
        int num_games = config.get<int>("games", 10, 0);
        bool use_omp = config.get<bool>("openmp", false, 0);
        int num_reads = config.get<int>("num_reads", 800, 0);
        float temperature = config.get<float>("temperature", 1.0f, 0);
        int temperature_cutoff = config.get<int>("temperature_cutoff", 30, 0);

        std::string save_path = std::filesystem::absolute(
            config.get<std::string>("save", "./datasets/all/", 0)
        ).string();

        MCTSSelfPlayConfig cfg(num_games, num_reads, temperature, temperature_cutoff);

        std::cout << "Dataset save path: " << save_path << std::endl;
        std::cout << "Use OpenMP: " << (use_omp ? "true" : "false") << std::endl;
        std::cout << "Number of CPUs: " << cpus << std::endl;
        std::cout << "Number of games: " << num_games << std::endl;
        std::cout << "Temperature: " << temperature << std::endl;
        std::cout << "Temperature cutoff: " << temperature_cutoff << std::endl;

        if (cpus > 1) {
            if (use_omp) {
#pragma omp parallel for
                for (int i = 0; i < cpus; ++i) {
                    MCTS_self_play(model_path, save_path, i, true, cfg);
                }
            }
            else {
                std::vector<std::thread> threads;
                for (int i = 0; i < cpus; ++i) {
                    threads.emplace_back(
                        MCTS_self_play,
                        model_path, save_path, i, true, cfg
                    );
                }

                // Дожидаемся завершения всех потоков
                for (auto& t : threads) {
                    if (t.joinable()) {
                        t.join();
                    }
                }
            }
        }
        else {
            MCTS_self_play(model_path, save_path, 0, false, cfg);
        }

    }
    else if (mode == "test") {
        int num_games = config.get<int>("games", 1, 0);
        int depth = config.get<int>("depth", 2, 0);
        int ai_side = config.get<int>("ai_side", 0, 0);
        int num_reads = config.get<int>("num_reads", 800, 0);
		bool is_native = config.get<bool>("is_native", 0, 0);
		bool load_weights = config.get<bool>("load_weights", false, 0);

        std::cout << "Number of games: " << num_games << std::endl;
        std::cout << "Depth: " << depth << std::endl;

		if (is_native) {
			std::cout << "Using native MCTS" << std::endl;
            try {
                self_play_native(model_path, num_games, depth, ai_side, num_reads, load_weights);
            }
			catch (const std::exception& e) {
				std::cerr << "Error: " << e.what() << std::endl;
				return 1;
			}
		}
		else {
			std::cout << "Using TorchScript MCTS" << std::endl;
            self_play(model_path, num_games, depth, ai_side, num_reads);
		}
    }
    else if (mode == "human") {
        int ai_side = config.get<int>("ai_side", 0, 0);
        int num_reads = config.get<int>("num_reads", 800, 0);
        play_against_alphazero(model_path, ai_side, num_reads);

    }
    else if (mode == "train") {
        int epochs = config.get<int>("epochs", 100, 0);
        double lr = config.get<double>("lr", 1e-4, 0);
        int lr_step = config.get<int>("lr_step", 100, 0);
        double gamma = config.get<double>("gamma", 100, 0);
        int batch_size = config.get<int>("batch_size", 32, 0);
		bool load_model = config.get<bool>("load_model", false, 0);
		bool load_weights = config.get<bool>("load_weights", false, 0);

		TrainConfig train_config(epochs, lr, lr_step, gamma, batch_size);

        std::string weights_path = std::filesystem::absolute(
            config.get<std::string>("save_weights", "./model_data/weights.dat", 0)
        ).string();
        std::string dataset_path = std::filesystem::absolute(
            config.get<std::string>("dataset", "./datasets/combined/combined_dataset.bin", 0)
        ).string();
        std::string save_path = std::filesystem::absolute(
            config.get<std::string>("save", "./model_data/best_optimized.pt", 0)
        ).string();

        std::cout << "Dataset path: " << dataset_path << std::endl;
        std::cout << "Model save path: " << save_path << std::endl;
        std::cout << "Weights save path: " << weights_path << std::endl;
        std::cout << "Epochs: " << epochs << std::endl;
        std::cout << "Batch size: " << batch_size << std::endl;
        std::cout << "Learning rate: " << lr << std::endl;
        std::cout << "Learning rate step: " << lr_step << std::endl;
        std::cout << "Learning rate step gamma: " << gamma << std::endl;
        std::cout << "Number of CPUs: " << cpus << std::endl;

        std::shared_ptr<TNET> model = std::make_shared<TNET>();
		if (load_model) {
			std::cout << "Loading model from: " << model_path << std::endl;
			try {
				torch::load(model, model_path);
			}
			catch (const c10::Error& e) {
				std::cerr << "Error loading model: " << e.what() << std::endl;
				return 1;
			}
        }
        else if (load_weights) {
            std::cout << "Loading model from: " << model_path << std::endl;
            try {
                model->load_weights(model_path);
            }
            catch (const std::exception& e) {
                std::cerr << "Error loading model: " << e.what() << std::endl;
                return 1;
            }
        }
        std::printf("Model loaded from %s\n", model_path.c_str());

        torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        std::printf("Using device: %s\n", device == torch::kCUDA ? "CUDA" : "CPU");

        model->to(device);
        model->train();
        if (epochs > 0)
            start_training(model, dataset_path, cpus, train_config);
		model->to(torch::kCPU);
		model->eval();
        torch::save(model, save_path);
        std::cout << "Model saved to " << save_path << std::endl;
		model->save_weights(weights_path);
		std::cout << "Model weights saved to " << weights_path << std::endl;
    }
    else {
        std::cerr << "Invalid mode: " << mode << std::endl;
        std::cerr << "Available modes: selfplay, test, train, human" << std::endl;
        return 1;
    }

    std::cout << "Execution finished." << std::endl;
    return 0;
}
