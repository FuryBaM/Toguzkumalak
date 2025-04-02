#include "pch.h"
#include "mcts_selfplay.h"

static void set_cpu_affinity(int cpu_id) {
    int num_cpus = std::thread::hardware_concurrency(); // Получаем количество доступных ядер
    int valid_cpu = cpu_id % num_cpus; // Гарантируем, что не выйдем за пределы

#ifdef _WIN32
    DWORD_PTR mask = 1ULL << valid_cpu;
    SetThreadAffinityMask(GetCurrentThread(), mask);
#endif

#ifndef _WIN32
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(valid_cpu, &cpuset);
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
#endif
}

std::string current_time() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);

    std::tm local_time;
#ifdef _WIN32
    if (localtime_s(&local_time, &now_time) != 0) {
#else
    if (localtime_r(&now_time, &local_time) == nullptr) {
#endif
        return "Error in getting local time";
    }

    std::ostringstream oss;
    oss << std::put_time(&local_time, "%H:%M:%S");
    return oss.str();
}

std::string elapsed_time(long long seconds) {
    int hours = seconds / 3600;
    int minutes = (seconds % 3600) / 60;
    int secs = seconds % 60;

    std::ostringstream oss;
    oss << std::setw(2) << std::setfill('0') << hours << ":"
        << std::setw(2) << std::setfill('0') << minutes << ":"
        << std::setw(2) << std::setfill('0') << secs;
    return oss.str();
}

std::string elapsed_time(float seconds) {
    return elapsed_time(static_cast<long long>(seconds));
}

std::string current_date() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);

    std::tm local_time;
#ifdef _WIN32
    if (localtime_s(&local_time, &now_time) != 0) {
#else
    if (localtime_r(&now_time, &local_time) == nullptr) {
#endif
        return "Error in getting local time";
    }

    std::ostringstream oss;
    oss << std::put_time(&local_time, "%Y-%m-%d");
    return oss.str();
}

std::string safe_filename(const std::string& save_path, int cpu, int i, const std::string& date) {
    std::filesystem::path path = save_path.empty() ? "./datasets/all" : save_path;

    std::string filename = "dataset_cpu" + std::to_string(cpu) + '_' +
        std::to_string(i) + '_' + date + ".bin";

    return (path / filename).string();
}

torch::jit::script::Module load_model(const std::string& model_path) {
    try {
        torch::jit::script::Module model;
        model = torch::jit::load(model_path);
        model.to(torch::kCPU);
        model.eval();
        std::cout << "Model loaded successfully!" << std::endl;
        return model;
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
}

std::pair<std::vector<float>, float> net_func(torch::jit::script::Module model, Game* game) {
    // Преобразуем состояние игры в тензор
    std::vector<float> game_state = game->toTensor(); // Реализуй toTensor() в Game
    // Вывод policy в формате [x1, x2, x3, ...]
    torch::Tensor input_tensor = torch::from_blob(game_state.data(), { 1, (2 * game->action_size) + 3 }).to(torch::kFloat);

    // Запускаем сеть
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);
    auto outputs = model.forward(inputs).toTuple();

    // Извлекаем policy и value
    at::Tensor policy_tensor = outputs->elements()[0].toTensor().contiguous();
    at::Tensor value_tensor = outputs->elements()[1].toTensor();

    // Конвертируем policy в std::vector<float>
    std::vector<float> child_priors(policy_tensor.data_ptr<float>(),
        policy_tensor.data_ptr<float>() + policy_tensor.numel());

    float value = value_tensor.item<float>();

    return { child_priors, value };
}

std::vector<float> softmax(const std::vector<float>& x) {
    std::vector<float> probabilities(x.size());
    float max_val = *std::max_element(x.begin(), x.end());

    float sum_exp = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        probabilities[i] = std::exp(x[i] - max_val);
        sum_exp += probabilities[i];
    }

    for (size_t i = 0; i < x.size(); ++i) {
        probabilities[i] /= sum_exp;
    }

    return probabilities;
}

std::vector<float> get_policy(UCTNode* root, float temperature) {
    std::vector<float> child_number_visits = root->child_number_visits;
    std::vector<float> log_visits(child_number_visits.size());

    for (size_t i = 0; i < child_number_visits.size(); ++i) {
        log_visits[i] = std::log(child_number_visits[i] + 1.0) / temperature;
    }

    return softmax(log_visits);
}

std::pair<int, std::vector<float>> UCT_search(torch::jit::script::Module model, Game* game, int num_reads, bool selfplay)
{
    std::vector<float> child_priors;
    float value_estimate;
    UCTNode* root = new UCTNode(new Game(*game), -1, new UCTNode(new Game(*game), -1, nullptr, selfplay, false), selfplay, true);
    for (int i = 0; i < num_reads; ++i)
    {
        UCTNode* leaf = root->select_leaf();
        Game* copied_game = leaf->game;
        std::pair<std::vector<float>, float> cv = net_func(model, copied_game);
        child_priors = cv.first;
        value_estimate = cv.second;
        if (game->checkWinner() != GAME_CONTINUE)
        {
            leaf->backup(value_estimate);
            continue;
        }
        leaf->expand(child_priors);
        leaf->backup(value_estimate);
    }
    std::vector<float> policy = get_policy(root);
    int action = argmax(root->child_number_visits);
    clearTree(root);
    return std::make_pair(action, policy);
}

void MCTS_self_play(std::string model_path, std::string save_path, int num_games, int cpu) {
    set_cpu_affinity(cpu);
    thread_local auto model = load_model(model_path);
    auto start_time = std::chrono::high_resolution_clock::now();

    printf("[%s] Process %d started\n", current_time().c_str(), cpu);

    for (int i = 0; i < num_games; i++) {
        Game game(9);
        GameState dataset;
        int value = 0;
        while (true) {
            int winner = game.checkWinner();
            std::string winner_name = "not finished";
            if (winner != GAME_CONTINUE || game.fullMoves >= 100) {
                std::string curr_time = current_time();
                if (winner == GAME_BLACK_WIN) {
                    value = -1;
                    winner_name = "black";
                }
                else if (winner == GAME_WHITE_WIN) {
                    value = 1;
                    winner_name = "white";
                }
                else {
                    value = 0;
                    winner_name = "draw";
                }
                auto end_time = std::chrono::high_resolution_clock::now();
                auto elapsed = elapsed_time(static_cast<long long>(
                    std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count()));
                printf("[%s][%s][%d]-Episode %d/%d Score: %d-%d Turns: %d Winner: %s\n",
                    curr_time.c_str(), elapsed.c_str(), cpu, i + 1, num_games,
                    game.player1_score, game.player2_score, game.fullMoves, winner_name.c_str());
                break;
            }
            auto result = UCT_search(model, &game, 800, true);
            int root_action = result.first;
            std::vector<float> policy = result.second;
            auto state = game.toTensor();
            float temperature = 1.0f;
            dataset.states.push_back(state);
            dataset.policies.push_back(policy);
            dataset.values.push_back(0);
            game.makeMove(root_action);
        }
        std::fill(dataset.values.begin() + 1, dataset.values.end(), value);
        std::string filename = safe_filename(save_path, cpu, i, current_date());
        dataset.save(filename);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed = elapsed_time(static_cast<long long>(
        std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count()));
    printf("[%s] Process %d finished. Elapsed: %s\n",
        current_time().c_str(), cpu, elapsed.c_str());
}

void self_play(std::string model_path, int num_games, int depth) {
    int white_wins = 0;
    int black_wins = 0;
    int ai_player = 0;
    auto model = load_model(model_path);
    auto start_time = std::chrono::high_resolution_clock::now();

    printf("[%s] Process started\n", current_time().c_str());
    for (int i = 0; i < num_games; ++i) {
        Game game(9);
        while (true) {
            game.showBoard();
            int action = 0;
            int winner = game.checkWinner();
            std::string winnerName = "not finished";
            if (winner != GAME_CONTINUE) {
                if (winner == GAME_WHITE_WIN) {
                    white_wins++;
                    winnerName = "white";
                }
                else if (winner == GAME_BLACK_WIN) {
                    black_wins++;
                    winnerName = "black";
                }
                else {
                    winnerName = "draw";
                }
                printf("[%s] Episode %d/%d AI: %d Score: %d-%d Moves: %d, Results: %d-%d Winner: %s\n",
                    current_time().c_str(), i + 1, num_games, ai_player,
                    game.player1_score, game.player2_score, game.fullMoves,
                    white_wins, black_wins, winnerName.c_str());
                break;
            }
            if (game.player == ai_player) {
                auto result = net_func(model, &game);
                std::vector<float> policy = result.first;
                float value = result.second;
                action = UCT_search(model, &game, 800, false).first;

                // Вывод policy в формате [x1, x2, x3, ...]
                printf("Policy: [");
                for (size_t i = 0; i < policy.size(); i++) {
                    printf("%f", policy[i]);
                    if (i < policy.size() - 1) {
                        printf(", ");
                    }
                }
                printf("]\n");

                printf("Value prediction: %f\n", value);
            }
            else {
                action = getMove(&game, depth);
            }
            auto game_state = game.toTensor();
            printf("State: [");
            for (size_t i = 0; i < game_state.size(); i++) {
                printf("%f", game_state[i]);
                if (i < game_state.size() - 1) {
                    printf(", ");
                }
            }
            printf("]\n");
            if (!game.makeMove(action)) {
                printf("Impossible move!\n");
            }
        }
    }
}
