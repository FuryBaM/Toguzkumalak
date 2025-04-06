#include "pch.h"
#include "mcts_selfplay.h"
#include "train.h"
#include "tnet.h"

int main(int argc, char** argv) {
    int num_games = 25;
    int cpus = std::thread::hardware_concurrency();
    omp_set_num_threads(cpus);
    std::string save_path = "";
	std::string dataset_path = "";
    std::string model_path = "";
    bool use_omp = false;
    std::string mode = "selfplay";
    int depth = 2;
    int ai_side = 0;
    int epochs = 100;

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
        else if (arg == "--omp") {
            use_omp = true;
        }
        else if (arg == "--depth" && i + 1 < argc) {
            depth = std::stoi(argv[++i]);
        }
        else if (arg == "--aiside" && i + 1 < argc) {
            ai_side = std::stoi(argv[++i]);
        }
        else if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        }
		else if (arg == "--dataset" && i + 1 < argc) {
			dataset_path = argv[++i];
		}
        else if (arg == "--epochs" && i + 1 < argc) {
            epochs = std::stoi(argv[++i]);
        }
    }

    if (save_path.empty()) {
        save_path = "./datasets/all/";
    }
    if (model_path.empty()) {
        model_path = "./model_data/best.pt";
    }
	if (dataset_path.empty()) {
		dataset_path = "./datasets/combined/combined_dataset.bin";
	}
	if (mode != "selfplay" && mode != "test" && mode != "human" && mode != "train") {
		std::cerr << "Invalid mode. Use 'selfplay', 'test', 'train' or 'human'." << std::endl;
		return 1;
	}
    save_path = std::filesystem::absolute(save_path).string();
    model_path = std::filesystem::absolute(model_path).string();
    dataset_path = std::filesystem::absolute(dataset_path).string();

    std::cout << "Mode: " << mode << std::endl;
    std::cout << "Model path: " << model_path << std::endl;

    if (mode == "selfplay") {
        std::cout << "Dataset save path: " << save_path << std::endl;
        std::cout << "Use open MP: " << (use_omp ? "true" : "false") << std::endl;
        std::cout << "Number of CPUs: " << cpus << std::endl;
        std::cout << "Number of games: " << num_games << std::endl;
        if (cpus > 1) {
            if (use_omp) {
#pragma omp parallel for
                for (int i = 0; i < cpus; ++i) {
                    MCTS_self_play(model_path, save_path, num_games, i);
                }
            }
            else {
                std::vector<std::future<void>> futures;
                for (int i = 0; i < cpus; ++i) {
                    futures.push_back(std::async(std::launch::async, MCTS_self_play, model_path, save_path, num_games, i, true));
                }
                for (auto& f : futures) {
                    f.get();
                }
            }
        }
		else {
			MCTS_self_play(model_path, save_path, num_games, 0, false);
		}
    }
    else if (mode == "test") {
        std::cout << "Number of games: " << num_games << std::endl;
        std::cout << "Depth: " << depth << std::endl;
        self_play(model_path, num_games, depth, ai_side);
    }
    else if (mode == "human") {
        play_against_alphazero(model_path, ai_side);
    }
	else if (mode == "train") {
        std::cout << "Dataset path: " << dataset_path << std::endl;
        std::cout << "Epochs: " << epochs << std::endl;
        std::cout << "Number of CPUs: " << cpus << std::endl;
        std::shared_ptr<TNET> model = std::make_shared<TNET>();
        torch::load(model, model_path);
		printf("Model loaded from %s\n", model_path.c_str());
        torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
		printf("Using device: %s\n", (device == torch::kCUDA ? "CUDA" : "CPU"));
        model->to(device);
        model->train();
		start_training(model, dataset_path, epochs, cpus);
	}
    else {
        std::cerr << "Unknown mode: " << mode << std::endl;
        return 1;
    }

    std::cout << "Execution finished." << std::endl;
    return 0;
}
