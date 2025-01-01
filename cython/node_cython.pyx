from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libc.math cimport sqrt
from libcpp cimport bool, nullptr, nullptr_t
from libcpp.memory cimport shared_ptr, make_shared
import numpy as np
cimport numpy as np
from game_cython cimport Game, minimax
from node_cython cimport UCTNode
cimport cython
np.import_array()

def minimaxPy(PyGame game, int player, int depth, float alpha, float beta):
    return minimax(game.thisptr, player, depth, alpha, beta)

cdef class PyGame:
    cdef Game *thisptr

    def __cinit__(self, int action_size):
        self.thisptr = new Game(action_size)

    def __dealloc__(self):
        del self.thisptr

    @property
    def player(self):
        return self.thisptr.player
    @player.setter
    def player(self, int value):
        self.thisptr.player = value

    @property
    def player1_score(self):
        return self.thisptr.player1_score
    @player1_score.setter
    def player1_score(self, int value):
        self.thisptr.player1_score = value

    @property
    def player2_score(self):
        return self.thisptr.player2_score
    @player2_score.setter
    def player2_score(self, int value):
        self.thisptr.player2_score = value

    @property
    def tuzdyk1(self):
        return self.thisptr.tuzdyk1
    @tuzdyk1.setter
    def tuzdyk1(self, int value):
        self.thisptr.tuzdyk1 = value

    @property
    def tuzdyk2(self):
        return self.thisptr.tuzdyk2
    @tuzdyk2.setter
    def tuzdyk2(self, int value):
        self.thisptr.tuzdyk2 = value
        
    @property
    def action_size(self):
        return self.thisptr.action_size
    @action_size.setter
    def action_size(self, int value):
        self.thisptr.action_size = value

    @property
    def max_stones(self):
        return self.thisptr.max_stones
    @max_stones.setter
    def max_stones(self, int value):
        self.thisptr.max_stones = value

    @property
    def goal(self):
        return self.thisptr.goal
    @goal.setter
    def goal(self, int value):
        self.thisptr.goal = value

    @property
    def semiMoves(self):
        return self.thisptr.semiMoves
    @semiMoves.setter
    def semiMoves(self, int value):
        self.thisptr.semiMoves = value

    @property
    def fullMoves(self):
        return self.thisptr.fullMoves
    @fullMoves.setter
    def fullMoves(self, int value):
        self.thisptr.fullMoves = value

    @property
    def lastMove(self):
        return self.thisptr.lastMove
    @lastMove.setter
    def lastMove(self, str value):
        self.thisptr.lastMove = value

    @property
    def boardArray(self):
        cdef vector[int] cpp_board = self.thisptr.boardArray
        return [cpp_board[i] for i in range(cpp_board.size())]
    @boardArray.setter
    def boardArray(self, list value):
        cdef vector[int] cpp_board = vector[int]()
        for v in value:
            cpp_board.push_back(v)
        self.thisptr.boardArray = cpp_board

    def set_action_size(self, int value):
        self.thisptr.setActionSize(value)

    def reset_game(self):
        self.thisptr.reset()

    @cython.wraparound(False)
    cpdef void show_board(self):
        cdef int i
        cdef str player1Row = ''
        cdef str player2Row = ''
        
        # Формируем строку для Player 1
        for i in range(0, self.action_size):
            if self.tuzdyk2 != i:
                player1Row += str(self.boardArray[i]) + " "
            else:
                player1Row += 'X '

        # Формируем строку для Player 2
        for i in range(self.action_size * 2 - 1, self.action_size - 1, -1):
            if self.tuzdyk1 != i:
                player2Row += str(self.boardArray[i]) + " "
            else:
                player2Row += 'X '

        # Выводим информацию на экран
        print('__________________________')
        print(f'Turn: {self.semiMoves}')
        print(f'Makes turn: {self.player}')
        print(f'Player 2 Score: {self.player2_score}')
        
        # Номера сверху (доска Player 2)
        print(' '.join([str(i) for i in range(self.action_size, 0, -1)]) + "\n")
        
        # Доска Player 2
        print(player2Row)
        
        # Доска Player 1
        print(player1Row + "\n")
        
        # Номера снизу (доска Player 1)
        print(' '.join([str(i) for i in range(1, self.action_size + 1)]))
        print(f'Player 1 Score: {self.player1_score}')
        print('__________________________')
        print(f'Last Move: {self.fullMoves}.{self.lastMove}')


    def evaluate_game(self,int player):
        return self.thisptr.evaluate(player)

    def get_player_moves_count(self, int player):
        return self.thisptr.getPlayerMovesCount(player)

    def get_player_row_stones(self, int player):
        return self.thisptr.getPlayerRowStones(player)

    def check_winner(self):
        return self.thisptr.checkWinner()

    def is_row_empty(self, int player):
        return self.thisptr.isRowEmpty(player)

    def is_valid_move(self, int x):
        return self.thisptr.isValidMove(x)

    def is_pit_empty(self, int x):
        return self.thisptr.isPitEmpty(x)

    def get_stone_count_at_pit(self, int x):
        return self.thisptr.getStoneCountAtPit(x)

    def get_stone_count_at_pit_2(self, int x, int y):
        return self.thisptr.getStoneCountAtPit(x, y)

    def get_possible_moves(self):
        return list(self.thisptr.getPossibleMoves())

    def switch_player(self):
        self.thisptr.switchPlayer()

    def make_move(self, int x):
        return self.thisptr.makeMove(x)

    def copy_game(self):
        cdef Game* copied_game = self.thisptr.copyGame()
        cdef PyGame py_game_copy = PyGame(copied_game.action_size)
        py_game_copy.thisptr = copied_game
        return py_game_copy

    def copy_board(self):
        return list(self.thisptr.copyBoard())

cdef public np.ndarray vector_to_numpy(list vec):
    cdef int n = len(vec)
    cdef np.ndarray[np.float32_t, ndim=1] result = np.zeros(n, dtype=np.float32)
    for i in range(n):
        result[i] = vec[i]
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float32_t, ndim=1] get_policy(UCTNode* root, float temperature=1.0):
    """
    Возвращает вероятности политики, рассчитанные на основе числа посещений.
    """
    cdef np.ndarray[np.float32_t, ndim=1] child_number_visits = vector_to_numpy(root.child_number_visits)
    cdef np.ndarray[np.float32_t, ndim=1] probabilities = softmax(
        1.0 / temperature * np.log(np.array(child_number_visits) + 1.0)
    )
    del root
    return probabilities

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float32_t, ndim=1] softmax(np.ndarray[np.float32_t, ndim=1] x):
    """
    Вычисляет softmax для входного вектора x.
    """
    cdef np.ndarray[np.float32_t, ndim=1] probabilities = np.exp(x - np.max(x))
    probabilities /= np.sum(probabilities)
    return probabilities

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float32_t, ndim=1] encodeBoard(PyGame game):
    """
    Кодирует текущее состояние игрового поля в вектор, подходящий для нейронной сети.
    """
    cdef int tuzdyk1 = game.tuzdyk1
    cdef int tuzdyk2 = game.tuzdyk2
    cdef int player = game.player
    cdef np.ndarray[np.float32_t, ndim=1] input_board = np.empty(
        shape=((game.action_size * 2) + 3), dtype=np.float32
    )
    cdef int i, j
    for i in range(2):
        for j in range(game.action_size):
            idx = (i * game.action_size) + j  # Индекс в одномерном массиве
            input_board[idx] = game.boardArray[i * game.action_size + j]  # Обращаемся к одномерному массиву board
            if i == 0 and tuzdyk2 == j:
                input_board[idx] = -1
            if i == 1 and tuzdyk1 == j:
                input_board[idx] = -1
    input_board[game.action_size * 2] = game.player1_score
    input_board[game.action_size * 2 + 1] = game.player2_score
    input_board[game.action_size * 2 + 2] = game.player
    return input_board

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef UCT_search(PyGame game, int num_reads, net_func, int self_play=1):
    """
    Реализация алгоритма UCT для выполнения поиска.
    """
    cdef int i
    cdef UCTNode* root
    cdef UCTNode* dummy
    cdef UCTNode* leaf
    cdef vector[float] child_priors
    cdef float value_estimate
    cdef PyGame py_game_copy

    # Создаём корневой узел дерева
    dummy = new UCTNode(game=game.thisptr, move=-1, parent=NULL, self_play=self_play)
    root = new UCTNode(game=game.thisptr, move=-1, parent=dummy, self_play=self_play)

    for i in range(num_reads):
        # Выбираем лист
        leaf = root.select_leaf()
        
        # Генерация child_priors и value_estimate через нейросетевую функцию
        py_game_copy = PyGame(game.thisptr.action_size)
        py_game_copy.thisptr = leaf.game
        child_priors, value_estimate = net_func(py_game_copy)
        
        # Если игра завершена, то возвращаем оценку обратно по дереву
        if leaf.game.checkWinner() != -1:
            leaf.backup(value_estimate)
            continue
        
        # Расширяем дерево и обновляем значения
        leaf.expand(np.array(child_priors, dtype=np.float32))
        leaf.backup(value_estimate)
    
    # Получаем политику перед удалением root
    cdef np.ndarray[np.float32_t, ndim=1] policy = get_policy(root)

    # Удаляем узлы дерева
    root.DestroyAllChildren()  # Рекурсивно удаляет все дочерние узлы
    del root
    del dummy

    return np.argmax(policy), policy