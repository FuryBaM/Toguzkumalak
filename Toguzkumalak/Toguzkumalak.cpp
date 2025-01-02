#include <iostream>
#include "game.h"
#include "node.h"
#include <bit>
#include <memory>

int getMove(Game* game, int depth)
{
    int bestmove = -1;
    float besteval = -std::numeric_limits<float>::infinity();
    int player = game->player;
    std::vector<int> actions = game->getPossibleMoves();
    for (int i = 0; i < actions.size(); ++i)
    {
        int move = actions[i];
        Game* game_copy = game->copyGamePtr();
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

std::pair<std::vector<float>, float> net(Game* game)
{
    // Генератор случайных чисел
    static std::random_device rd;
    static std::mt19937 gen(rd());  // Генератор случайных чисел
    static std::uniform_real_distribution<> dis(-1.0, 1.0); // Диапазон от 0 до 1

    // Генерация случайных значений для вектора
    std::vector<float> random_values(9);
    for (int i = 0; i < 9; ++i)
    {
        random_values[i] = static_cast<float>(dis(gen)); // Генерация значения от 0 до 1
    }

    // Генерация случайного значения для второго числа
    float random_value = static_cast<float>(dis(gen));

    return std::pair<std::vector<float>, float>(random_values, random_value);
}

std::pair<float, UCTNode> UCT_search(Game* game, int num_reads, std::pair<std::vector<float>, float>(*net_func)(Game*), bool selfplay)
{
    std::vector<float> child_priors;
    float value_estimate;
    UCTNode dummy = UCTNode(game, -1, nullptr, selfplay);
    UCTNode root = UCTNode(game, -1, &dummy, selfplay);
    for (int i = 0; i < num_reads; ++i)
    {
        UCTNode* leaf = root.select_leaf();
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
    root.destroyAllChildren();
    std::cout << "Child visits: ";
    for (int v : root.child_number_visits) std::cout << v << " ";
    std::cout << std::endl;
    return std::pair<float, UCTNode>((float)argmax(root.child_number_visits), root);
}

int main()
{
    int move;
    Game* game = new Game(9);
    game->showBoard();
    while (game->checkWinner() == GAME_CONTINUE)
    {
        //move = getMove(game, 10);
        std::pair<float, UCTNode> result = UCT_search(game, 800, net, true);
        std::cout << "Children " << result.second.children.size() << std::endl;
        move = result.first;
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
}