#include "pch.h"

#include "game.h"
#include "node.h"

torch::jit::script::Module model;

int getMove(Game* game, int depth)
{
    int bestmove = -1;
    float besteval = -std::numeric_limits<float>::infinity();
    int player = game->player;
    std::vector<int> actions = game->getPossibleMoves();
    for (int i = 0; i < actions.size(); ++i)
    {
        int move = actions[i];
        Game* game_copy = new Game(*game);
        game_copy->makeMove(move);
        float eval = minimax(game_copy, player, depth - 1, -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());
        if (eval > besteval)
        {
            besteval = eval;
            bestmove = move;
        }
        delete game_copy;
    }
    return bestmove;
}

void load_model(const std::string& model_path) {
    try {
        model = torch::jit::load(model_path);
        model.to(torch::kCPU);
        model.eval();
        std::cout << "Model loaded successfully!" << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
}

std::pair<std::vector<float>, float> net_func(Game* game) {
    // Преобразуем состояние игры в тензор
    std::vector<float> game_state = game->toTensor(); // Реализуй toTensor() в Game
    auto input_tensor = torch::from_blob(game_state.data(), { 1, static_cast<int64_t>(game_state.size()) }).to(torch::kFloat);

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

std::vector<float> get_policy(UCTNode* root, float temperature = 1.0) {
    std::vector<float> child_number_visits = root->child_number_visits;
    std::vector<float> log_visits(child_number_visits.size());

    for (size_t i = 0; i < child_number_visits.size(); ++i) {
        log_visits[i] = std::log(child_number_visits[i] + 1.0) / temperature;
    }

    return softmax(log_visits);
}

std::pair<float, std::vector<float>> UCT_search(Game* game, int num_reads, std::pair<std::vector<float>, float>(*net_func)(Game*), bool selfplay)
{
    std::vector<float> child_priors;
    float value_estimate;
    Game* copy = new Game(*game);
    UCTNode* root = new UCTNode(copy, -1, new UCTNode(new Game(*copy), -1, nullptr, selfplay, false), selfplay, true);
    for (int i = 0; i < num_reads; ++i)
    {
        UCTNode* leaf = root->select_leaf();
        Game* copied_game = leaf->game;
        std::pair<std::vector<float>, float> cv = net_func(copied_game);
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
    float action = argmax(root->child_number_visits);
    clearTree(root);
    return std::make_pair(action, policy);
}

int main()
{
    load_model("current_trained_net2.pt");
    int move;
    Game* game = new Game(9);
    game->showBoard();
    while (game->checkWinner() == GAME_CONTINUE)
    {
        if (game->player == -1)
        {
            move = getMove(game, 6);
        }
        else
        {
            std::pair<float, std::vector<float>> result = UCT_search(game, 800, net_func, false);
            move = result.first;
        }
        game->makeMove(move);
        game->showBoard();
    }
    std::string win_msg = "Unknown";
    switch (game->checkWinner())
    {
    case GAME_DRAW:
        win_msg = "Draw!";
        break;
    case GAME_WHITE_WIN:
        win_msg = "White wins!";
        break;
    case GAME_BLACK_WIN:
        win_msg = "Black wins!";
        break;
    case GAME_CONTINUE:
        win_msg = "Game is not finished.";
        break;
    default:
        break;
    }
    std::cout << win_msg << std::endl;
    delete game;
    _CrtDumpMemoryLeaks();
}