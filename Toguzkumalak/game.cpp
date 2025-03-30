#include "pch.h"

#include "game.h"

Game::Game(int a_size)
{
    setActionSize(a_size);
    reset();
}

Game::Game(const Game& game)
{
    size_t size = game.action_size * 2;
    setActionSize(game.action_size);

    boardArray = new int[size];
    std::memcpy(boardArray, game.boardArray, size * sizeof(int));

    player = game.player;
    player1_score = game.player1_score;
    player2_score = game.player2_score;
    tuzdyk1 = game.tuzdyk1;
    tuzdyk2 = game.tuzdyk2;
    semiMoves = game.semiMoves;
    fullMoves = game.fullMoves;
    lastMove = game.lastMove;
}

Game::~Game()
{
    delete[] boardArray;
}

void Game::setActionSize(int a_size)
{
    action_size = std::clamp(a_size, 2, 100);
    action_size = a_size;
    max_stones = a_size * a_size * 2;
    goal = a_size * a_size + 1;
}


void Game::reset()
{
    size_t size = action_size * 2;
    boardArray = new int[size];
    std::fill(boardArray, boardArray + size, action_size);
    player = player1_score = player2_score = 0;
    tuzdyk1 = tuzdyk2 = -1;
    semiMoves = fullMoves = 0;
    lastMove = "";
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
    std::cout << "Last Move: " << fullMoves << "." << lastMove << "\n";
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
    else if (player1_score >= goal || (player2_score < goal && isRowEmpty(BLACK)))
    {
        return GAME_WHITE_WIN;
    }
    else if (player2_score >= goal || (player1_score < goal && isRowEmpty(WHITE)))
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
        return boardArray[x] == 0;
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
            moves.push_back(i % action_size);
        }
    }
    return moves;
}

void Game::switchPlayer()
{
    player = 1 - player;
}
bool Game::makeMove(int x)
{
    x += (action_size * player);
    if (!isValidMove(x) || checkWinner() != GAME_CONTINUE)
    {
        return false;
    }
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
    int board_size = action_size * 2;
    for (int i = 0; i < stonesInArm; ++i)
    {
        x = (x + 1) % (board_size);
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
            boardArray[x] = 0;
        }
        else if (player == BLACK && 0 <= x && x < action_size)
        {
            player2_score += boardArray[x];
            boardArray[x] = 0;
        }
    }
    fullMoves += 1 - player;
    semiMoves++;
    switchPlayer();
    lastMove += std::to_string(x % action_size + 1);
    return true;
}

Game::board Game::copyBoard()
{
    size_t size = action_size * 2;
    int* copy = new int[size];
    std::memcpy(copy, boardArray, size * sizeof(board));
    return copy;
}

std::vector<float> Game::toTensor() {
    std::vector<float> input_board((action_size * 2) + 3, 0.0f);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < action_size; ++j) {
            int idx = (i * action_size) + j;
            input_board[idx] = static_cast<float>(boardArray[idx]);

            if (i == 0 && tuzdyk2 == j) {
                input_board[idx] = -1.0f;
            }
            if (i == 1 && tuzdyk1 == j) {
                input_board[idx] = -1.0f;
            }
        }
    }

    input_board[action_size * 2] = static_cast<float>(player1_score);
    input_board[action_size * 2 + 1] = static_cast<float>(player2_score);
    input_board[action_size * 2 + 2] = static_cast<float>(player);

    return input_board;
}

float Game::evaluate(int player)
{
    int half = action_size / 2;
    int winner = checkWinner();
    float win_factor = 0.0f;
    float score_factor = 0.0f;
    float tuzdyk_factor = 0.0f;
    float possible_moves_factor = getPlayerMovesCount(player);
    float row_stones_factor = getPlayerRowStones(player) - getPlayerRowStones(1 - player);
    if (goal == 0 || max_stones == 0) {
        return 0;
    }

    if (player == WHITE)
    {
        score_factor = player1_score - player2_score;
        if (winner == WHITE)
        {
            win_factor = max_stones - player1_score;
        }
        else if (winner == BLACK)
        {
            win_factor = player1_score - max_stones;
        }
        if (tuzdyk1 != -1)
        {
            tuzdyk_factor += (half - std::abs(tuzdyk1 - half)) * half;
        }
        if (tuzdyk2 != -1)
        {
            tuzdyk_factor -= (half - std::abs(tuzdyk2 - half)) * half;
        }
    }
    else if (player == BLACK)
    {
        score_factor = player2_score - player1_score;
        if (winner == WHITE)
        {
            win_factor = max_stones - player2_score;
        }
        else if (winner == BLACK)
        {
            win_factor = player2_score - max_stones;
        }
        if (tuzdyk2 != -1)
        {
            tuzdyk_factor += (half - std::abs(tuzdyk2 - half)) * half;
        }
        if (tuzdyk1 != -1)
        {
            tuzdyk_factor -= (half - std::abs(tuzdyk1 - half)) * half;
        }
    }

    if (winner == GAME_DRAW)
    {
        win_factor += max_stones / action_size;
    }

    float score = score_factor / goal +
        row_stones_factor / max_stones +
        tuzdyk_factor / goal +
        possible_moves_factor / action_size;
    float symbol = score >= 0 ? 1.0f : -1.0f;
    float eval_result = symbol * std::sqrt(std::abs(score)) + (win_factor / goal);

    return eval_result;
}

float minimax(Game* game, int player, int depth, float alpha, float beta)
{
    float eval = 0;
    float max_eval = -std::numeric_limits<float>::infinity();
    float min_eval = std::numeric_limits<float>::infinity();

    if (depth <= 0 || game->checkWinner() != GAME_CONTINUE)
    {
        float eval = game->evaluate(player);
        return eval;
    }

    std::vector<int> actions = game->getPossibleMoves();

    if (game->player == player)
    {
        for (int i = 0; i < actions.size(); ++i)
        {
            int move = actions[i];
            Game game_copy = Game(*game);
            game_copy.makeMove(move);
            eval = minimax(&game_copy, player, depth - 1, alpha, beta);
            max_eval = std::max(max_eval, eval);
            alpha = std::max(alpha, eval);
            if (beta <= alpha)
            {
                break;
            }
        }
        return max_eval;
    }
    else
    {
        for (int i = 0; i < actions.size(); ++i)
        {
            int move = actions[i];
            Game game_copy = Game(*game);
            game_copy.makeMove(move);
            eval = minimax(&game_copy, player, depth - 1, alpha, beta);
            min_eval = std::min(min_eval, eval);
            beta = std::min(beta, eval);
            if (beta <= alpha)
            {
                break;
            }
        }
        return min_eval;
    }
}

int getMove(Game* game, int depth)
{
    int bestmove = -1;
    float besteval = -std::numeric_limits<float>::infinity();
    int player = game->player;
    std::vector<int> actions = game->getPossibleMoves();
    for (int i = 0; i < actions.size(); ++i)
    {
        int move = actions[i];
        Game* game_copy = new Game(*game);
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

void clearGame(Game* game) {
    if (game) {
        delete game;
    }
}