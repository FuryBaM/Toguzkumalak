#include "pch.h"

#include "node.h"

UCTNode::UCTNode(Game* game, int move, UCTNode* parent, bool selfplay, bool is_root)
{
	this->game = game;
	this->move = move;
	this->parent = parent;
	this->self_play = selfplay;
	this->is_root = is_root;
	is_expanded = false;
	action_size = game->action_size;
	children = {};
	if (parent) 
	{
		child_priors = std::vector<float>(action_size, 0.0f);
		child_total_value = std::vector<float>(action_size, 0.0f);
		child_number_visits = std::vector<float>(action_size, 0.0f);
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

void UCTNode::destroyAllChildren() {
	for (auto& child : children) {
		delete child.second;
	}
	children.clear();
}

UCTNode::~UCTNode() {
	destroyAllChildren();
	delete game;
}

std::vector<float> UCTNode::child_Q() {
	std::vector<float> Q(action_size);
	for (size_t i = 0; i < Q.size(); ++i) {
		Q[i] = this->child_total_value[i] / (1.0f + this->child_number_visits[i]);
	}
	return Q;
}

std::vector<float> UCTNode::child_U() {
	std::vector<float> U(action_size);
	float sqrt_visits = std::sqrt((this->getNumberVisits()));
	for (size_t i = 0; i < U.size(); ++i) {
		U[i] = sqrt_visits * (std::abs(this->child_priors[i]) / (1.0f + this->child_number_visits[i]));
	}
	return U;
}

int UCTNode::best_child()
{
	std::vector<float> Q = this->child_Q();
	std::vector<float> U = this->child_U();
	std::vector<float> combined(Q.size());

	for (size_t i = 0; i < Q.size(); ++i) {
		combined[i] = Q[i] + U[i];
	}
	if (this->action_idxes.empty() == false)
	{
		float max_value = std::numeric_limits<float>::lowest();
		int best_action = -1;

		for (int idx : action_idxes) {
			if (combined[idx] > max_value) {
				max_value = combined[idx];
				best_action = idx;
			}
		}
		return best_action;
	}
	else
	{
		int max_id = 0;
		for (int i = 0; i < combined.size(); ++i)
		{
			if (combined[i] > combined[max_id])
			{
				max_id = i;
			}
		}
		return max_id;
	}
}

UCTNode* UCTNode::select_leaf()
{
	UCTNode* current = this;
	int bestmove = -1;
	while (current->is_expanded == true)
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

	std::vector<float> dirichlet_noise = generate_dirichlet_noise(valid_child_priors.size(), this->a);

	for (size_t i = 0; i < valid_child_priors.size(); ++i) {
		valid_child_priors[i] = 0.75f * valid_child_priors[i] + 0.25f * dirichlet_noise[i];
	}

	for (size_t i = 0; i < action_idxs.size(); ++i) {
		child_priors[action_idxs[i]] = valid_child_priors[i];
	}

	return child_priors;
}

void UCTNode::expand(std::vector<float> child_priors_)
{
	this->is_expanded = true;
	std::vector<int> action_idxs;
	std::vector<float> c_p = child_priors_;
	action_idxs = game->getPossibleMoves();
	if (action_idxs.size() == 0)
	{
		this->is_expanded = false;
	}
	action_idxes = action_idxs;
	for (int i = 0; i < child_priors_.size(); ++i)
	{
		if (contains(action_idxs, i) == false)
		{
			c_p[i] = 0.0f;
		}
	}
	if (is_root && self_play)
	{
		c_p = this->add_dirichlet_noise(action_idxs, c_p);
	}
	this->child_priors = c_p;
}

UCTNode* UCTNode::try_add_child(int move)
{
	if (contains(children, move) == false)
	{
		Game* copy_game = new Game(*game);
		copy_game->makeMove(move);
		UCTNode* parent = this;
		this->children[move] = new UCTNode(copy_game, move, parent, self_play, false);
	}
	return children[move];
}

void UCTNode::backup(float value_estimate)
{
	UCTNode* current = this;
	while (current->parent != nullptr)
	{
		current->setNumberVisits(1.0f + current->getNumberVisits());
		if (current->game->player == BLACK)
		{
			current->setTotalValue(current->getTotalValue() + (1.0f * value_estimate));
		}
		else if (current->game->player == WHITE)
		{
			current->setTotalValue(current->getTotalValue() + (-1.0f * value_estimate));
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

std::vector<float> generate_dirichlet_noise(size_t size, float alpha) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::gamma_distribution<float> dist(alpha, 1.0f);

	std::vector<float> noise(size);
	float sum = 0.0f;
	for (size_t i = 0; i < size; ++i) {
		noise[i] = dist(gen);
		sum += noise[i];
	}

	for (size_t i = 0; i < size; ++i) {
		noise[i] /= sum;
	}

	return noise;
}

void clearTree(UCTNode* root) {
	if (root) {
		delete root->parent;
		delete root;
	}
}
