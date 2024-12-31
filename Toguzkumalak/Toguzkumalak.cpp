#include <iostream>
#include "game.h"
#include "node.h"
#include <memory>

int getMove(Game * game, int depth)
{
    int bestmove = -1;
    float besteval = -std::numeric_limits<float>::infinity();
    int player = game->player;
    std::vector<int> actions = game->getPossibleMoves();
    for (int i = 0; i < actions.size(); ++i)
    {
        int move = actions[i];
        Game* game_copy = game->copyGame();
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
    return std::pair<std::vector<float>, float>(std::vector<float>{ 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f }, 0.1f);
}

std::pair<float, UCTNode*> UCT_search(Game* game, int num_reads, std::pair<std::vector<float>, float>(*net_func)(Game*), bool selfplay)
{
    UCTNode* root;
    UCTNode* leaf;
    std::vector<float> child_priors;
    float value_estimate;
    root = new UCTNode(game, -1, new UCTNode(game, -1, nullptr, selfplay), selfplay);
    Game* copied_game;
    for (int i = 0; i < num_reads; ++i)
    {
        leaf = root->select_leaf();
        copied_game = leaf->game;
        std::pair<std::vector<float>, float> cv = net_func(game);
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
    return std::pair<float, UCTNode*>((float)argmax(root->child_number_visits), root);
}

int main()
{
    int move;
    Game game;
    game.showBoard();
    while (game.checkWinner() == GAME_CONTINUE)
    {
        //move = getMove(&game, 2);
        std::pair<float, UCTNode*> result = UCT_search(game.copyGame(), 800, net, false);
        move = result.first;
        game.makeMove(move);
        game.showBoard();
        delete result.second;
    }
    std::string win_msg = "Unknown";
    switch (game.checkWinner())
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
}