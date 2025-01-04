#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
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
    static std::uniform_real_distribution<float> dis1(-1.0f, 1.0f); // Диапазон от -1 до 1
    static std::uniform_real_distribution<float> dis2(0.0f, 1.0f); // Диапазон от 0 до 1

    // Генерация случайных значений для вектора
    std::vector<float> random_values(9);
    for (int i = 0; i < 9; ++i)
    {
        random_values[i] = static_cast<float>(dis2(gen)); // Генерация значения от -1 до 1
    }

    // Генерация случайного значения для второго числа
    float random_value = static_cast<float>(dis1(gen));

    auto defined = std::make_pair(std::vector<float>{0.0373f, 0.1493f, 0.0746f, 0.0896f, 0.0448f, 0.0075f, 0.2239f, 0.1791f, 0.1939f}, +0.1f);
    auto random = std::make_pair(random_values, random_value);
    return defined;
}

std::pair<float, UCTNode*> UCT_search(Game* game, int num_reads, std::pair<std::vector<float>, float>(*net_func)(Game*), bool selfplay)
{
    std::vector<float> child_priors;
    float value_estimate;
    Game* copy = game->copyGamePtr();
    UCTNode* root = new UCTNode(copy, -1, new UCTNode(copy->copyGamePtr(), -1, nullptr, selfplay, false), selfplay, true);
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
    std::cout << "Child visits: ";
    for (float v : root->child_number_visits) std::cout << v << " ";
    std::cout << std::endl;
    std::cout << "Child priors: ";
    for (float v : root->child_priors) std::cout << v << " ";
    std::cout << std::endl;
    std::cout << "Child total: ";
    for (float v : root->child_total_value) std::cout << v << " ";
    std::cout << std::endl;
    return std::make_pair(argmax(root->child_number_visits), root);
}

int main()
{
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    int move;
    Game* game = new Game(9);
    game->showBoard();
    while (game->checkWinner() == GAME_CONTINUE)
    {
        if (game->player == -1)
        {
            move = getMove(game,6);
        }
        else
        {
            std::pair<float, UCTNode*> result = UCT_search(game, 100, net, false);
            std::cout << "Children " << result.second->children.size() << std::endl;
            move = result.first;
            delete result.second->parent;
            delete result.second;
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
    _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_DEBUG);
    _CrtDumpMemoryLeaks();
}