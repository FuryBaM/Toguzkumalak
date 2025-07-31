import math
import gc

import numpy as np
from mcts import PyGame, UCT_search, minimaxPy, encodeBoard

import numpy as np

def net_func(game):
    """
    Возвращает случайные child_priors и value_estimate для тестирования.
    """
    # Генерация случайных вероятностей для child_priors
    child_priors = np.random.random(9)  # 9 действий, значения от 0 до 1
    child_priors /= child_priors.sum()  # Нормализация, чтобы сумма была равна 1

    # Генерация случайного значения value_estimate
    value_estimate = np.random.random() * 2 - 1  # Значение от -1 до 1

    return np.float32(child_priors), value_estimate

def getMove(game, depth = 8):
    best_move = -1
    best_eval = -math.inf
    player = game.player
    for move in game.get_possible_moves():
        game_copy = game.copy_game()
        game_copy.make_move(move)
        eval = minimaxPy(game_copy, player, depth - 1, -math.inf, math.inf)
        if eval > best_eval:
            best_move = move
            best_eval = eval
    return best_move

# game = PyGame(9)
# game_copy = game.copy_game()
# while game.check_winner() == -1:
#     game.show_board()
#     game.make_move(getMove(game, 1))
# game.show_board()-
for i in range(1000):
    game = PyGame(9)
    while game.check_winner() == -1:
        game.show_board()
        res, policy = UCT_search(game, 800, net_func, False)
        print("Result", res, policy)
        game.make_move(res)
    game.show_board()
