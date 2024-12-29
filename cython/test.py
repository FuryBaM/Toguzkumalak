import math

import numpy as np
from mcts import UCT_search, PyGame, encodeBoard

def getMove(game, depth = 2):
    best_move = -1
    best_eval = -math.inf
    player = game.player
    for move in game.get_possible_moves():
        game_copy = game.copy_game()
        game_copy.make_move(move)
        eval = minimaxPy(game, player, depth - 1, -math.inf, math.inf)
        if eval > best_eval:
            best_move = move
            best_eval = eval
    return best_move

game = PyGame(9)
# while game.check_winner() == -1:
#     game.show_board()
#     game.make_move(getMove(game, 4))
# game.show_board()

print(encodeBoard(game))

def net_func(game):
    return np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32), 0.5
print(net_func(game))
res = UCT_search(game, 800, net_func, False)
print(res[0])