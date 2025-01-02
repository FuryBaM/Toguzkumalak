#include "node.h"
#include <stdexcept>
#include <limits>
#include <bit>
#include <memory>
#include <iostream>


UCTNode::UCTNode(Game* game, int move, UCTNode* parent, bool selfplay, bool is_root) : game(game), move(move), parent(parent), self_play(selfplay), is_root(is_root)
{
	this->game = game;
	this->move = move;
	this->parent = parent;
	this->self_play = selfplay;
	this->is_root = is_root;
	is_expanded = 0;
	action_size = game->action_size;
	children = {};
	if (parent) 
	{
		child_priors = std::vector<float>(action_size, 0);
		child_total_value = std::vector<float>(action_size, 0);
		child_number_visits = std::vector<float>(action_size, 0);
	}
	else if (parent == nullptr)
	{
		child_priors = {};
		child_total_value = {};
		child_number_visits = {};
	}
	action_idxes = std::vector<int>();
	a = 10.0f / action_size;
}

void UCTNode::destroyChildren()
{
	children.clear();
}

void UCTNode::destroyAllChildren()
{
	for (auto child : children)
	{
		if (child.second)
		{
			child.second->destroyAllChildren();
			delete child.second;
		}
	}
}

UCTNode::~UCTNode(){
	if (is_root)
	{
		destroyAllChildren();
	}
	delete game;
}

std::vector<float> UCTNode::child_Q() {
	std::vector<float> Q(child_total_value.size());
	for (size_t i = 0; i < Q.size(); ++i) {
		Q[i] = child_total_value[i] / (1 + child_number_visits[i]);
	}
	return Q;
}

std::vector<float> UCTNode::child_U() {
	std::vector<float> U(child_total_value.size());
	float sqrt_visits = std::sqrt((getNumberVisits()));
	for (size_t i = 0; i < U.size(); ++i) {
		U[i] = sqrt_visits * std::abs(child_priors[i]) / (1 + child_number_visits[i]);
	}
	return U;
}

int UCTNode::best_child()
{
	auto Q = child_Q();
	auto U = child_U();
	std::vector<float> combined(Q.size());

	for (size_t i = 0; i < Q.size(); ++i) {
		combined[i] = Q[i] + U[i];
	}
	if (!action_idxes.empty())
	{
		int best_idx = action_idxes[0];
		float max_value = combined[action_idxes[0]];

		for (int idx : action_idxes) {
			if (combined[idx] > max_value) {
				max_value = combined[idx];
				best_idx = idx;
			}
		}
		return best_idx;
	}
	else
	{
		auto max_it = std::max_element(combined.begin(), combined.end());
		return std::distance(combined.begin(), max_it);
	}
}

UCTNode* UCTNode::select_leaf()
{
	UCTNode* current = this;
	int bestmove;
	while (current->is_expanded)
	{
		bestmove = current->best_child();
		current = current->try_add_child(bestmove);
	}
	return current;
}

std::vector<float> UCTNode::add_dirichlet_noise(std::vector<int> action_idxs, std::vector<float> child_priors) {
	std::vector<float> valid_child_priors(action_idxs.size());

	for (size_t i = 0; i < action_idxs.size(); ++i) {
		valid_child_priors[i] = child_priors[action_idxs[i]];
	}

	std::vector<float> dirichlet_noise = generate_dirichlet_noise(valid_child_priors.size());

	for (size_t i = 0; i < valid_child_priors.size(); ++i) {
		valid_child_priors[i] = 0.75f * valid_child_priors[i] + 0.25f * dirichlet_noise[i];
	}

	for (size_t i = 0; i < action_idxs.size(); ++i) {
		child_priors[action_idxs[i]] = valid_child_priors[i];
	}

	return child_priors;
}

void UCTNode::expand(std::vector<float> child_priors)
{
	is_expanded = true;
	std::vector<int> action_idxs;
	std::vector<float> c_p = child_priors;
	Game::board possible_actions = game->getPossibleMoves();
	for (auto action : possible_actions)
	{
		action_idxs.push_back(action);
	}
	if (action_idxs.size() == 0)
	{
		is_expanded = false;
	}
	action_idxes = action_idxs;
	for (int i = 0; i < child_priors.size(); ++i)
	{
		if (!contains(action_idxs, i))
		{
			c_p[i] = 0;
		}
	}
	if (this->parent->parent == nullptr && self_play)
	{
		c_p = this->add_dirichlet_noise(action_idxs, c_p);
	}
	child_priors = c_p;
}

UCTNode* UCTNode::try_add_child(int move)
{
	if (!contains(children, move))
	{
		Game* copy_game = game->copyGamePtr();
		copy_game->makeMove(move);
		UCTNode* parent = this;
		children[move] = new UCTNode(copy_game, move, parent, self_play, false);
	}
	return children[move];
}

void UCTNode::backup(float value_estimate)
{
	UCTNode* current = this;
	while (current->parent != nullptr)
	{
		current->setNumberVisits(current->getNumberVisits() + 1);
		if (current->game->player == BLACK)
		{
			current->setTotalValue(current->getTotalValue() + (1 * value_estimate));
		}
		else if (current->game->player == WHITE)
		{
			current->setTotalValue(current->getTotalValue() + (-1 * value_estimate));
		}
		current = current->parent;
	}
}

int argmax(const std::vector<float>& vec) {
	if (vec.empty()) {
		throw std::invalid_argument("Input vector is empty");
	}

	return std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));
}

std::vector<float> generate_dirichlet_noise(size_t size) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::gamma_distribution<float> dist(1.0f, 1.0f); // ��������� ��� �����-�������������

	std::vector<float> noise(size);
	float sum = 0.0f;
	for (size_t i = 0; i < size; ++i) {
		noise[i] = dist(gen); // ��������� ���������� �������� ��� �������
		sum += noise[i];
	}

	// ������������, ����� ����� ���� ����� 1
	for (size_t i = 0; i < size; ++i) {
		noise[i] /= sum;
	}

	return noise;
}