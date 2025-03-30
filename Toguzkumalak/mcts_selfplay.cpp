#include "pch.h"
#include "mcts_selfplay.h"

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
    Game* copy = new Game(*game);
    UCTNode* root = new UCTNode(copy, -1, new UCTNode(new Game(*copy), -1, nullptr, selfplay, false), selfplay, true);
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

void MCTS_self_play(std::string model_path, int num_games, int cpu) {
    auto model = load_model(model_path);
    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "[" << current_time() << "] Process " << std::to_string(cpu) << " started" << "\n";

    for (int i = 0; i < num_games; i++) {
        Game* game = new Game(9);
        GameState dataset;
        int value = 0;
        while (true) {
            int winner = game->checkWinner();
            std::string winner_name = "not finished";
            if (winner != GAME_CONTINUE || game->fullMoves >= 100) {
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
                std::cout << "[" << curr_time << "]" << "[" << elapsed << "]" <<
                    "[" << cpu << "]-Episode " << i + 1 << "/" << std::to_string(num_games) << " " <<
                    "Score: " << std::to_string(game->player1_score) << "-" << std::to_string(game->player2_score) << " " <<
                    "Turns: " << std::to_string(game->fullMoves) << " " <<
                    "Winner: " << winner_name << "\n";
                break;
            }
            auto result = UCT_search(model, new Game(*game), 800, true);
            int root_action = result.first;
            std::vector<float> policy = result.second;
            auto state = game->toTensor();
            float temperature = 1.0f;
            dataset.states.push_back(state);
            dataset.policies.push_back(policy);
            dataset.values.push_back(0);
            game->makeMove(root_action);
        }
        for (int i = 1; i < dataset.values.size(); ++i) {
            dataset.values[i] = value;
        }
        std::string filename = "dataset_cpu" + std::to_string(cpu) + '_' + std::to_string(i) + '_' + current_date() + ".bin";
        dataset.save(filename);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed = elapsed_time(static_cast<long long>(
        std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count()));
    std::cout << "[" << current_time() << "] Process " << std::to_string(cpu) << " finished. Elapsed: " << elapsed << "\n";
}
