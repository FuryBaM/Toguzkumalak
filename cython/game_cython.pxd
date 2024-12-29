from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.stdlib cimport malloc
from libc.stdlib cimport free
from libcpp cimport bool
cdef extern from "game.h":
    float minimax(Game* game, int player, int depth, float alpha, float beta)
    cdef cppclass Game:
        int player
        int player1_score
        int player2_score
        int tuzdyk1
        int tuzdyk2
        int semiMoves
        int fullMoves
        string lastMove
        vector[int] boardArray
        int action_size
        int max_stones
        int goal

        Game(int action_size)
        void setActionSize(int value)
        void reset()
        void showBoard()
        float evaluate(int player)
        int getPlayerMovesCount(int player)
        int getPlayerRowStones(int player)
        int checkWinner()
        bool isRowEmpty(int player)
        bool isValidMove(int x)
        bool isPitEmpty(int x)
        int getStoneCountAtPit(int x)
        int getStoneCountAtPit(int x, int y)
        vector[int] getPossibleMoves()
        void switchPlayer()
        int makeMove(int x)
        Game* copyGame()
        vector[int] copyBoard()