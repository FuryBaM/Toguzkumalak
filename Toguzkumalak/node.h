#pragma once
#include "game.h"
#include <unordered_map>
class UCTNode
{
public:
	Game* game;
	int move;
	bool isExpanded;
	bool selfplay;
	UCTNode* parent;
	std::unordered_map<int, UCTNode*> children;
	std::vector<float> child_priors;
	std::vector<float> child_total_value;
	std::vector<float> child_total_visits;
	std::vector<int> action_idxes;
	float a;
	UCTNode(Game* game, int move = -1, UCTNode* parent = nullptr, bool selfplay = false);
	~UCTNode();
	float getNumberVisits();
	void setNumberVisits(float value);
	float getTotalValue();
	void setTotalValue(float value);
	std::vector<float> child_Q();
	std::vector<float> child_U();
	int best_child();
	UCTNode* select_leaf();
	std::vector<float> add_dirichlet_noise(std::vector<float> action_idxs, std::vector<float> child_priors);
	void expand(std::vector<float> child_priors);
	UCTNode* try_add_child(int move);
	void backup(float value_estimate);
};

