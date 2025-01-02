#pragma once

#include <vector>
#include <array>
#include <string>
#include <algorithm>
#include <cmath>

#define WHITE 0
#define BLACK 1

#define GAME_CONTINUE -1
#define GAME_WHITE_WIN 0
#define GAME_BLACK_WIN 1
#define GAME_DRAW 2

class Game
{
private:
	bool isPlayer(int player) { return (player != 0 && player != 1); }
public:
	using board = std::vector<int>;
	int action_size;
	int max_stones;
	int goal;
	board boardArray;
	int player;
	int player1_score;
	int player2_score;
	int tuzdyk1;
	int tuzdyk2;
	int semiMoves;
	int fullMoves;
	std::string lastMove;

	Game(int action_size = 9);
	~Game() = default;
	void setActionSize(int value);
	void reset();
	void showBoard();
	float evaluate(int player);
	int getPlayerMovesCount(int player);
	int getPlayerRowStones(int player);
	int checkWinner();
	bool isRowEmpty(int player);
	bool isValidMove(int x);
	bool isPitEmpty(int x);
	int getStoneCountAtPit(int x);
	int getStoneCountAtPit(int x, int y);
	std::vector<int> getPossibleMoves();
	void switchPlayer();
	bool makeMove(int x);
	Game copyGame();
	Game* copyGamePtr();
	std::vector<int> copyBoard();
};
float minimax(Game* game, int player, int depth, float alpha, float beta);