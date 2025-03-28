#pragma once
#include "game.h"
#include <unordered_map>

class Node 
{
public:
	Game* game;
	size_t action_size = 9;
	int move;
	bool is_root;
	Node* parent;
	std::vector<Node*> children;
	float visits;
	float value;
	float rchild_n = 0.0f;
	float rchild_t = 0.0f;
	std::vector<int> action_idxes;

	bool isFullyExpanded();
	Node* bestChild(float explorationWeight = 1.0f);
	void expand(Node* node);
	void backPropagate(Node* node, float reward);
	void simulate(Node* node);
};

