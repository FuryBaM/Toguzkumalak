#pragma once
#include "game.h"

class UCTNode
{
public:
	Game* game;
	size_t action_size = 9;
	int move;
	bool is_expanded;
	bool self_play;
	bool is_root;
	UCTNode* parent;
	std::unordered_map<int, UCTNode*> children;
	std::vector<float> child_priors;
	std::vector<float> child_total_value;
	std::vector<float> child_number_visits;
	float rchild_n = 0.0f;
	float rchild_t = 0.0f;
	std::vector<int> action_idxes;
	float a = 0.0f;

	UCTNode(Game* game, int move, UCTNode* parent, bool selfplay, bool is_root);
	void destroyAllChildren();
	~UCTNode();
	float getNumberVisits()
	{
		if (move == -1) return rchild_n;
		return this->parent->child_number_visits[this->move];
	}
	void setNumberVisits(float value)
	{
		if (move == -1)
		{
			rchild_n = value;
			return;
		}
		this->parent->child_number_visits[this->move] = value;
	}
	float getTotalValue()
	{
		if (move == -1) return rchild_t;
		return this->parent->child_total_value[this->move];
	}
	void setTotalValue(float value)
	{
		if (move == -1)
		{
			rchild_t = value;
			return;
		}
		this->parent->child_total_value[this->move] = value;
	}
	std::vector<float> child_Q();
	std::vector<float> child_U();
	int best_child();
	UCTNode* select_leaf();
	std::vector<float> add_dirichlet_noise(std::vector<int> action_idxs, std::vector<float> child_priors);
	void expand(std::vector<float> child_priors);
	UCTNode* try_add_child(int move);
	void backup(float value_estimate);
};

template<typename T>
bool contains(const std::vector<T>& vec, const T& value) {
	return std::find(vec.begin(), vec.end(), value) != vec.end();
}
template<typename K, typename V>
bool contains(const std::unordered_map<K, V>& map, const K& key) {
	return map.find(key) != map.end();
}

int argmax(const std::vector<float>& vec);
std::vector<float> generate_dirichlet_noise(size_t size, float alpha);
void clearTree(UCTNode* node);