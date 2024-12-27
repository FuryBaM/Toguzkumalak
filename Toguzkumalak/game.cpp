#include "game.h"
#include <iostream>

Game::Game(int a_size)
{
	setActionSize(a_size);
	reset();
}

void Game::reset()
{
	size_t size = action_size * 2;
	boardArray = std::vector<int>(size, action_size);
	player = player1_score = player2_score = 0;
	tuzdyk1 = tuzdyk2 = -1;
	semiMoves = fullMoves = 1;
	lastMove = "";
}

void Game::setActionSize(int a_size)
{
	if (a_size <= 1)
	{
		a_size = 9;
	}
	action_size = a_size;
    max_stones = a_size * a_size * 2;
    goal = a_size * a_size + 1;
}

void Game::showBoard() {
    std::string player1Row = "";
    std::string player2Row = "";

    // Формирование строки для Player 1
    for (int i = 0; i < action_size; ++i) {
        if (tuzdyk2 != i) {
            player1Row += std::to_string(boardArray[i]) + " ";
        }
        else {
            player1Row += "X ";
        }
    }

    // Формирование строки для Player 2
    for (int i = action_size * 2 - 1; i >= action_size; --i) {
        if (tuzdyk1 != i) {
            player2Row += std::to_string(boardArray[i]) + " ";
        }
        else {
            player2Row += "X ";
        }
    }

    // Вывод на экран
    std::cout << "__________________________\n";
    std::cout << "Move: " << semiMoves << "\n";
    std::cout << "Makes move: " << player << "\n";
    std::cout << "Player 2 Score: " << player2_score << "\n";

    // Номера сверху (доска Player 2)
    for (int i = action_size; i > 0; --i) {
        std::cout << i << " ";
    }
    std::cout << "\n\n";

    // Доска Player 2
    std::cout << player2Row << "\n";

    // Доска Player 1
    std::cout << player1Row << "\n\n";

    // Номера снизу (доска Player 1)
    for (int i = 1; i <= action_size; ++i) {
        std::cout << i << " ";
    }
    std::cout << "\n";

    std::cout << "Player 1 Score: " << player1_score << "\n";
    std::cout << "__________________________\n";
    std::cout << "Last Move: " << lastMove << "\n";
}

float Game::evaluate(int player)
{
    return 0;
}

int Game::getPlayerMovesCount(int player)
{
    int totalMoves = 0;
    int start = action_size * player;
    int end = action_size + (action_size * player);
    for (int i = start; i < end; ++i)
    {
        if (!getStoneCountAtPit(i))
        {
            totalMoves++;
        }
    }
    return totalMoves;
}

int Game::getPlayerRowStones(int player)
{
    int stones = 0;
    int start = action_size * player;
    int end = action_size + (action_size * player);
    for (int i = start; i < end; ++i)
    {
        stones += getStoneCountAtPit(i);
    }
    return stones;
}

int Game::checkWinner()
{
    if (player1_score == (goal - 1) && player2_score == (goal - 1))
    {
        return GAME_DRAW;
    }
    else if (player1_score >= goal || player2_score < goal && isRowEmpty(BLACK))
    {
        return GAME_WHITE_WIN;
    }
    else if (player2_score >= goal || player1_score < goal && isRowEmpty(WHITE))
    {
        return GAME_BLACK_WIN;
    }
    return GAME_CONTINUE;
}

bool Game::isRowEmpty(int player)
{
    int start = action_size * player;
    int end = action_size + (action_size * player);
    for (int i = start; i < end; ++i)
    {
        if (getStoneCountAtPit(i)) return false;
    }
    return true;
}

bool Game::isValidMove(int x)
{
    return ((player == WHITE && 0 <= x && x < action_size) ||
        (player == BLACK && action_size <= x && x < action_size * 2)) &&
        !isPitEmpty(x);
}

bool Game::isPitEmpty(int x)
{
    if (0 <= x && x < action_size * 2) {
        return boardArray[x] == 0;  // Яма пуста, если значение равно 0
    }
    throw std::out_of_range("Pit index out of range");
}

int Game::getStoneCountAtPit(int x)
{
    if (x < 0 || x >= action_size * 2)
    {
        throw std::out_of_range("Received cell is out of range");
    }
    return boardArray[x];
}

int Game::getStoneCountAtPit(int x, int y)
{
    if (x < 0 || x >= action_size * 2 || (y != 0 && y != 1))
    {
        throw std::out_of_range("Received cell is out of range");
    }
    return boardArray[x + (action_size * y)];
}

std::vector<int> Game::getPossibleMoves()
{
    std::vector<int> moves;
    if (checkWinner() != GAME_CONTINUE)
    {
        return moves;
    }
    int start = action_size * player;
    int end = action_size + (action_size * player);
    for (int i = start; i < end; ++i)
    {
        if (!isPitEmpty(i))
        {
            moves.push_back(i);
        }
    }
}

void Game::switchPlayer()
{
    player = 1 - player;
}
bool Game::makeMove(int x)
{
    if (!isValidMove(x) || checkWinner() != GAME_CONTINUE)
    {
        return false;
    }
    int aPlayer = x < action_size ? WHITE : BLACK;
    int stonesInArm = 0;
    lastMove = std::to_string((x % action_size + 1));
    if (getStoneCountAtPit(x) > 1)
    {
        stonesInArm = getStoneCountAtPit(x) - 1;
        boardArray[x] = 1;
    }
    else
    {
        stonesInArm = 1;
        boardArray[x] = 0;
    }
    for (int i = 0; i < stonesInArm; ++i)
    {
        x = (x + 1) % (action_size * 2);
        boardArray[x]++;
    }
    if (tuzdyk1 != -1)
    {
        player1_score += getStoneCountAtPit(tuzdyk1);
        boardArray[tuzdyk1] = 0;
    }
    if (tuzdyk2 != -1)
    {
        player2_score += getStoneCountAtPit(tuzdyk2);
        boardArray[tuzdyk2] = 0;
    }
    if (boardArray[x] == 3 && x != action_size * player - 1)
    {
        if (player == WHITE && action_size <= x && x < action_size * 2 && tuzdyk1 == -1)
        {
            if (tuzdyk2 != x % action_size)
            {
                tuzdyk1 = x;
                boardArray[x] = 0;
                player1_score += 3;
            }
        }
        else if (player == BLACK && 0 <= x && x < action_size && tuzdyk2 == -1)
        {
            if (tuzdyk1 != x % action_size)
            {
                tuzdyk2 = x;
                boardArray[x] = 0;
                player2_score += 3;
            }
        }
    }
    if (boardArray[x] % 2 == 0)
    {
        if (player == WHITE && action_size <= x && x < action_size * 2)
        {
            player1_score += boardArray[x];
        }
        else if (player == BLACK && 0 <= x && x < action_size)
        {
            player1_score += boardArray[x];
        }
    }
    fullMoves += player;
    switchPlayer();
    lastMove += std::to_string(x % action_size + 1);
    return true;
}