from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libc.math cimport sqrt
from libcpp cimport bool, nullptr, nullptr_t
from libcpp.memory cimport shared_ptr, make_shared
import numpy as np
cimport numpy as np
from game_cython cimport Game, minimax
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

cdef class UCTNode:
    cdef public PyGame game
    cdef public int action_size
    cdef public int move
    cdef public int is_expanded
    cdef public int self_play
    cdef public UCTNode parent
    cdef public dict children
    cdef public float [:] child_priors
    cdef public float [:] child_total_value
    cdef public float [:] child_number_visits
    cdef public list action_idxes
    cdef public float a

    def __cinit__(self, PyGame game, int move = -1, UCTNode parent=None, int self_play = 1):
        self.game = game
        self.action_size = game.action_size
        self.move = move
        self.is_expanded = 0
        self.self_play = self_play
        self.parent = parent
        self.children = {}
        self.child_priors = np.zeros(self.action_size, dtype=np.float32)
        self.child_total_value = np.zeros(self.action_size, dtype=np.float32)
        self.child_number_visits = np.zeros(self.action_size, dtype=np.float32)
        self.action_idxes = []
        self.a = 10.0/self.action_size

    property number_visits:
        def __get__(self):
            return self.parent.child_number_visits[self.move]
        def __set__(self, float value):
            self.parent.child_number_visits[self.move] = value

    property total_value:
        def __get__(self):
            return self.parent.child_total_value[self.move]
        def __set__(self, float value):
            self.parent.child_total_value[self.move] = value
            
    @cython.wraparound(False)
    cdef np.ndarray[np.float32_t, ndim = 1] child_Q(self):
        return np.array(self.child_total_value) / (1 + np.array(self.child_number_visits))
        
    @cython.wraparound(False)
    cdef np.ndarray[np.float32_t, ndim = 1] child_U(self):
        return sqrt(self.number_visits) * (abs(np.array(self.child_priors)) / (1 + np.array(self.child_number_visits)))
        
    @cython.wraparound(False)
    cdef int best_child(self):
        cdef np.ndarray[np.float32_t, ndim = 1] q_plus_u = self.child_Q() + self.child_U()
        cdef np.ndarray[np.float32_t, ndim = 1] actions
        cdef int bestmove
        cdef int i, idx
        if len(self.action_idxes) != 0:
            bestmove = self.action_idxes[np.argmax(q_plus_u[self.action_idxes])]
        else:
            bestmove = np.argmax(q_plus_u)
        return bestmove
        
    @cython.wraparound(False)
    cdef UCTNode select_leaf(self):
        cdef UCTNode current = self
        cdef int best_move = 0
        while current.is_expanded:
            best_move = current.best_child()
            current = current.maybe_add_child(best_move)
        return current
        
    @cython.wraparound(False)
    cdef np.ndarray[np.float32_t, ndim = 1] add_dirichlet_noise(self, list action_idxs, np.ndarray[np.float32_t, ndim = 1] child_priors):
        cdef np.ndarray[np.float32_t, ndim = 1] valid_child_priors = child_priors[action_idxs]
        cdef int i, idx
        valid_child_priors = np.zeros(len(action_idxs), dtype=np.float32)
        cdef np.ndarray[np.float32_t, ndim = 1] dirichlet_noise = np.random.dirichlet(np.zeros(len(valid_child_priors), dtype=np.float32) + self.a).astype(np.float32)
        valid_child_priors = 0.75 * valid_child_priors + 0.25 * dirichlet_noise
        child_priors[action_idxs] = valid_child_priors
        return child_priors
        
    @cython.wraparound(False)
    cdef void expand(self, np.ndarray[np.float32_t, ndim = 1] child_priors):
        self.is_expanded = 1
        cdef int i
        cdef list action_idxs = []
        cdef np.ndarray[np.float32_t, ndim = 1] c_p = child_priors
        cdef list possible_actions = self.game.get_possible_moves()
        for action in possible_actions:
            action_idxs.append(action)
        if len(action_idxs) == 0:
            self.is_expanded = 0
        self.action_idxes = action_idxs
        for i in range(len(child_priors)):
            if i not in action_idxs:
                c_p[i] = 0.0000000000
        if self.parent.parent is None and self.self_play:
            c_p = self.add_dirichlet_noise(action_idxs, c_p)
        self.child_priors = c_p
        
    @cython.wraparound(False)
    cdef UCTNode maybe_add_child(self, int move):
        if move not in self.children:
            copy_board = self.game.copyGame()
            copy_board.make_move(move)
            self.children[move] = UCTNode(copy_board, move, self, self.self_play)
        return self.children[move]
        
    @cython.wraparound(False)
    cdef backup(self, float value_estimate):
        cdef UCTNode current = self
        while current.parent is not None:
            current.number_visits += 1
            if current.game.player == 1:
                current.total_value += (1 * value_estimate)
            elif current.game.player == 0:
                current.total_value += (-1 * value_estimate)
            current = current.parent

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float32_t, ndim=1] get_policy(UCTNode root, float temperature=1.0):
    """
    Возвращает вероятности политики, рассчитанные на основе числа посещений.
    """
    cdef np.ndarray[np.float32_t, ndim=1] probabilities = softmax(
        1.0 / temperature * np.log(np.array(root.child_number_visits) + 1e-10)
    )
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
    cdef UCTNode root
    cdef UCTNode leaf
    cdef vector[float] child_priors
    cdef float value_estimate

    # Создаём корневой узел дерева
    root = UCTNode(game, -1, None, self_play)
    cdef PyGame py_game_copy
    
    for i in range(num_reads):
        # Выбираем лист
        leaf = root.select_leaf()
        # Генерация child_priors и value_estimate через нейросетевую функцию
        py_game_copy = leaf.game.copy_game()
        child_priors, value_estimate = net_func(py_game_copy)
        # Если игра завершена, то возвращаем оценку обратно по дереву
        if leaf.game.check_winner() != -1:
            print("backup start")
            leaf.backup(value_estimate)
            print("backup end")
            continue
        # Расширяем дерево и обновляем значения
        print("check start expand")
        leaf.expand(np.array(child_priors, dtype=np.float32))
        print("check1")
        leaf.backup(value_estimate)
        print("check2")
    print("check-end")
    # Возвращаем лучшую найденную политику и корень дерева
    return np.argmax(root.child_number_visits), root