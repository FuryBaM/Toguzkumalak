# distutils: language=c++
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=1

from libc.math cimport sqrt
import numpy as np
cimport numpy as np
cimport cython
np.import_array()

cdef class UCTNode:
    cdef public Game game
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

    def __cinit__(self, Game game, int move = -1, UCTNode parent=None, int self_play = 1):
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
        cdef list possible_actions = self.game.getPossibleMoves()
        for action in possible_actions:
            action_idxs.append(action[0])
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
            copy_board.makeMove(move, copy_board.player)
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
            
@cython.wraparound(False)
cpdef np.ndarray[np.float32_t, ndim = 1] get_policy(UCTNode root, float temperature=1.0):
    cdef np.ndarray[np.float32_t, ndim = 1] probabilities = softmax(1.0 / temperature * np.log(np.array(root.child_number_visits) + 1e-10))
    return probabilities
    
@cython.wraparound(False)
cdef np.ndarray[np.float32_t, ndim = 1] softmax(np.ndarray[np.float32_t, ndim = 1] x):
    cdef np.ndarray[np.float32_t, ndim = 1] probabilities = np.exp(x - np.max(x))
    probabilities /= np.sum(probabilities)
    return probabilities

cpdef np.ndarray[np.float32_t, ndim=1] encodeBoard(Game game):
    cdef int tuzdyk1 = game.tuzdyk1
    cdef int tuzdyk2 = game.tuzdyk2
    cdef int player = game.player
    cdef np.ndarray[np.float32_t, ndim=1] input_board = np.empty(shape=((game.action_size * 2) + 3), dtype=np.float32)
    cdef int i, j
    cdef int player_score, player_side
    for i in range(2):
        for j in range(game.action_size):
            idx = (i*(game.action_size-1))+j
            input_board[idx] = game.board[i][j]
            if i == 0 and tuzdyk2 == j:
                input_board[idx] = -1
            if i == 1 and tuzdyk1 == j:
                input_board[idx] = -1
    input_board[18] = game.player1_score
    input_board[19] = game.player2_score
    input_board[20] = game.player
    return input_board

cpdef UCT_search(Game game, int num_reads, net_func, int self_play = 1):
    cdef int i
    cdef UCTNode root
    cdef UCTNode leaf
    cdef np.ndarray[np.float32_t, ndim=1] child_priors = np.empty(game.action_size, dtype = np.float32)
    cdef float value_estimate
    root = UCTNode(game, -1, UCTNode(game, -1, None, self_play), self_play)
    for i in range(num_reads):
        leaf = root.select_leaf()
        child_priors, value_estimate = net_func(leaf.game)
        if leaf.game.checkWinner() != -1:
            leaf.backup(value_estimate)
            continue
        leaf.expand(child_priors)
        leaf.backup(value_estimate)
    
    return np.argmax(root.child_number_visits), root

cdef class Game:
    cdef public int action_size
    cdef public int max_stones
    cdef public int goal
    cdef public list board
    cdef public int player
    cdef public int player1_score
    cdef public int player2_score
    cdef public int tuzdyk1
    cdef public int tuzdyk2
    cdef public int semiTurns
    cdef public int fullTurns
    cdef public str lastMove

    def __cinit__(self, int action_size = 9):
        self.setActionSize(action_size)
        self.board = [[self.action_size] * self.action_size for _ in range(2)]
        self.player = 0
        self.player1_score = 0
        self.player2_score = 0
        self.tuzdyk1 = -1
        self.tuzdyk2 = -1
        self.semiTurns = 1
        self.fullTurns = 1
        self.lastMove = ''
        
    @cython.wraparound(False)
    cpdef void reset(self):
        self.board = [[self.action_size] * self.action_size for _ in range(2)]
        self.player1_score = self.player2_score = 0
        self.player = 0
        self.tuzdyk1 = self.tuzdyk2 = -1
        self.semiTurns = self.fullTurns = 1
        self.lastMove = ''
        
    @cython.wraparound(False)
    cdef void setActionSize(self, int value):
        self.action_size = value
        self.max_stones = (value**2) * 2
        self.goal = (value**2) + 1
        
    @cython.wraparound(False)
    cpdef void showBoard(self):
        cdef int i
        cdef str player1Desk = ''
        for i in range(self.action_size):
            if self.tuzdyk2 != i:
                player1Desk += str(self.board[0][i]) + " "
            else:
                player1Desk += 'X '

        cdef str player2Desk = ""
        for i in range(self.action_size - 1, -1, -1):
            if self.tuzdyk1 != i:
                player2Desk += str(self.board[1][i]) + " "
            else:
                player2Desk += 'X '

        print('__________________________\n')
        print(f'Turn: {self.semiTurns}\nMakes turn: {self.player}')
        print(f'Player 2 Score: {self.player2_score}')
        print(''.join([f"{i} " for i in (range(self.action_size, 0, -1))]) + "\n")
        print(player2Desk)
        print(player1Desk + "\n")
        print(''.join([f'{i} ' for i in range(1, self.action_size + 1)]))
        print(f'Player 1 Score: {self.player1_score}\n')
        print('__________________________')
        print(f'Last Move: {self.lastMove}')
        
    @cython.wraparound(False)
    cpdef float evaluate(self, int player):
        cdef float half = self.action_size // 2
        cdef int winner = self.checkWinner()
        cdef float win_factor = 0
        cdef float score_factor = 0
        cdef float tuzdyk_factor = 0
        cdef float possible_moves_factor = self.getPlayerMovesCount(player)
        cdef float desk_stones_factor = self.getPlayerDeskStones(player) - self.getPlayerDeskStones(1 - player)
        if player == 0:
            score_factor = self.player1_score - self.player2_score
            if winner == 0:
                win_factor = self.max_stones - self.player1_score
            elif winner == 1:
                win_factor = self.player1_score - self.max_stones
            if self.tuzdyk1 != -1:
                tuzdyk_factor += (half - abs(self.tuzdyk1 - half)) * half
            if self.tuzdyk2 != -1:
                tuzdyk_factor -= (half - abs(self.tuzdyk2 - half)) * half
        elif player == 1:
            score_factor = self.player2_score - self.player1_score
            if winner == 1:
                win_factor = self.max_stones - self.player2_score
            elif winner == 0:
                win_factor = self.player2_score - self.max_stones
            if self.tuzdyk2 != -1:
                tuzdyk_factor += (half - abs(self.tuzdyk2 - half)) * half
            if self.tuzdyk1 != -1:
                tuzdyk_factor -= (half - abs(self.tuzdyk1 - half)) * half
        if winner == 2:
            win_factor += self.max_stones // self.action_size
        cdef float eval_result = sqrt(score_factor * 1/self.goal + 
            desk_stones_factor * 1/self.max_stones + 
            tuzdyk_factor/self.goal +
            possible_moves_factor * 1/self.action_size)/4 + (win_factor/self.goal)
        return eval_result
        
    @cython.wraparound(False)
    cdef int getPlayerMovesCount(self, int player):
        cdef int totalMoves = 0
        for x in range(self.action_size):
            if self.getStoneCountAtCell(x, player) != 0:
                totalMoves += 1
        return totalMoves
        
    @cython.wraparound(False)
    cdef int getPossibleMovesCount(self):
        cdef int totalMoves = 0
        for x in range(self.action_size):
            if self.getStoneCountAtCell(x, self.player) != 0:
                totalMoves += 1
        return totalMoves
        
    @cython.wraparound(False)
    cdef int getPlayerDeskStones(self, int player):
        cdef int result = 0
        for i in range(self.action_size):
            result += self.getStoneCountAtCell(i, player)
        return result
        
    @cython.wraparound(False)
    cpdef int checkWinner(self):
        if self.player1_score == self.goal - 1 and self.player2_score == self.goal - 1: # draw
            return 2
        elif self.player1_score >= self.goal or (
                self.player2_score < self.goal and self.isDeskEmpty(1)
        ): # white wins
            return 0
        elif self.player2_score >= self.goal or (
                self.player1_score < self.goal and self.isDeskEmpty(0)
        ): # black wins
            return 1
        return -1
        
    @cython.wraparound(False)
    cdef int isDeskEmpty(self, int player):
        for i in range(self.action_size):
            if self.board[player][i] > 0:
                return 0
        return 1
    @cython.wraparound(False)
    cdef int isValidMove(self, int x, int y):
        return (
                (self.player == 0 and y == 0)
                or (self.player == 1 and y == 1)
        ) and self.isCellEmpty(x, y) == False
    @cython.wraparound(False)
    cdef int isCellEmpty(self, int x, int y):
        return self.board[y][x] == 0
    @cython.wraparound(False)
    cdef int getStoneCountAtCell(self, int x, int y):
        return self.board[y][x]
        
    @cython.wraparound(False)
    cpdef list getPossibleMoves(self):
        cdef list possibleMoves = []
        if self.checkWinner() != -1:
            return possibleMoves
        for x in range(self.action_size):
            if not self.isCellEmpty(x, self.player):
                possibleMoves.append((x, self.player))
        return possibleMoves
        
    @cython.wraparound(False)
    cdef void switchPlayer(self):
        self.player = 1 - self.player
        
    @cython.wraparound(False)
    cpdef int makeMove(self, int x, int y):
        if not self.isValidMove(x, y) or self.checkWinner() != -1:
            return 0
        cdef int stonesInArm = 0
        cdef int i = 0
        self.lastMove = str(x + 1)
        self.semiTurns += 1
        if self.getStoneCountAtCell(x, y) > 1:
            stonesInArm = self.getStoneCountAtCell(x, y) - 1
            self.board[y][x] = 1
        else:
            stonesInArm = 1
            self.board[y][x] = 0
        for _ in range(stonesInArm):
            y ^= (x == (self.action_size - 1))
            x = (x + 1) % self.action_size
            self.board[y][x] += 1
        if self.tuzdyk1 != -1:
            self.player1_score += self.getStoneCountAtCell(self.tuzdyk1, 1)
            self.board[1][self.tuzdyk1] = 0
        if self.tuzdyk2 != -1:
            self.player2_score += self.getStoneCountAtCell(self.tuzdyk2, 0)
            self.board[0][self.tuzdyk2] = 0
        if self.board[y][x] == 3 and x != self.action_size - 1:  # make tuzdyk
            if (self.player == 0 and y == 1 and self.tuzdyk1 == -1):
                if self.tuzdyk2 != x:
                    self.tuzdyk1 = x
                    self.board[y][x] = 0  # player 1
                    self.player1_score += 3
            elif (self.player == 1 and y == 0 and self.tuzdyk2 == -1):
                if self.tuzdyk1 != x:
                    self.tuzdyk2 = x
                    self.board[y][x] = 0  # player 2
                    self.player2_score += 3
        if self.board[y][x] % 2 == 0 and self.player != y:  # grab stones
            if self.player == 0:
                self.player1_score += self.board[y][x]
            elif self.player == 1:
                self.player2_score += self.board[y][x]
            self.board[y][x] = 0
        self.fullTurns += self.player
        self.switchPlayer()
        self.lastMove += str(x + 1) + " "
        return 1
        
    @cython.wraparound(False)
    cpdef Game copyGame(self):
        game = Game(self.action_size)
        game.player1_score = self.player1_score
        game.player2_score = self.player2_score
        game.board = self.copyBoard(self.board)
        game.player = self.player
        game.tuzdyk1 = self.tuzdyk1
        game.tuzdyk2 = self.tuzdyk2
        game.semiTurns = self.semiTurns
        game.fullTurns = self.fullTurns
        game.lastMove = self.lastMove
        return game
        
    @cython.wraparound(False)
    cpdef list copyBoard(self, list boardArray):
        cdef list copiedBoard = []
        for row in boardArray:
            copiedBoard.append(row[:])
        return copiedBoard
        
@cython.wraparound(False)
cpdef float minimax(Game game, int player, int depth, float alpha, float beta):
    cdef float max_eval
    cdef float min_eval
    cdef float eval
    cdef Game copy_game
    if depth == 0 or game.checkWinner() != -1:
        return game.evaluate(player)
    elif game.player == player:
        max_eval = -1000
        for move in game.getPossibleMoves():
            game_copy = game.copyGame()
            game_copy.makeMove(move[0], move[1])
            eval = minimax(game_copy, player, depth - 1, alpha, beta)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    elif game.player == 1 - player:
        min_eval = 1000
        for move in game.getPossibleMoves():
            game_copy = game.copyGame()
            game_copy.makeMove(move[0], move[1])
            eval = minimax(game_copy, player, depth - 1, alpha, beta)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval