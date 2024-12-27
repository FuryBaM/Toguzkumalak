#include <iostream>
#include "game.h"
int main()
{
    int move;
    Game game;
    game.showBoard();
    while (game.checkWinner() == GAME_CONTINUE)
    {
        std::cin >> move;
        game.makeMove(move + (game.action_size * game.player - 1));
        game.showBoard();
    }
}