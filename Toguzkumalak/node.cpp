#include "node.h"
#include <stdexcept>
#include <limits>
#include <memory>


UCTNode::UCTNode(Game* game, int move, UCTNode* parent, bool selfplay)
{
	this->game = game;
	this->move = move;
	this->parent = parent;
	this->self_play = selfplay;
	is_expanded = 0;
	action_size = game->action_size;
	children = std::unordered_map<int, UCTNode*>();
	child_priors = std::vector<float>(action_size, 0);
	child_total_value = std::vector<float>(action_size, 0);
	child_number_visits = std::vector<float>(action_size, 0);
	action_idxes = std::vector<int>();
	a = 10.0 / action_size;
}

UCTNode::~UCTNode() = default;

std::vector<float> UCTNode::child_Q()
{
	return vectorDivide(child_total_value, child_number_visits);
}

std::vector<float> UCTNode::child_U()
{
	std::vector<float> child_num_visits = addNumberToVector(child_number_visits, 1.0f);
	return multiplyNumberToVector(vectorDivide(absVector(child_priors), child_num_visits), static_cast<float>(sqrt(getNumberVisits())));
}

int UCTNode::best_child()
{
	std::vector<float> q_plus_u = vectorAdd(child_Q(), child_U());
	int bestmove;
	if (action_idxes.size() != 0)
	{
		bestmove = action_idxes[argmax(q_plus_u, action_idxes)];
	}
	else
	{
		bestmove = argmax(q_plus_u);
	}
	return bestmove;
}

UCTNode* UCTNode::select_leaf()
{
	UCTNode* current = this;
	int bestmove = 0;
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
	if (parent != nullptr && parent->parent == nullptr && self_play)
	{
		c_p = add_dirichlet_noise(action_idxs, c_p);
	}
	child_priors = c_p;
}

UCTNode* UCTNode::try_add_child(int move)
{
	if (!contains(children, move))
	{
		Game* copy_game = game->copyGame();
		copy_game->makeMove(move);
		children[move] = new UCTNode(copy_game, move, this, self_play);
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

// Функция для нахождения индекса максимального значения среди определенных индексов
int argmax(const std::vector<float>& vec, const std::vector<int>& indices) {
	if (indices.empty()) {
		throw std::invalid_argument("Indices vector is empty");
	}

	float max_val = vec[indices[0]];
	int best_idx = indices[0];
	for (int idx : indices) {
		if (vec[idx] > max_val) {
			max_val = vec[idx];
			best_idx = idx;
		}
	}

	return best_idx;
}
std::vector<float> generate_dirichlet_noise(size_t size) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::gamma_distribution<float> dist(1.0f, 1.0f); // Параметры для гамма-распределения

	std::vector<float> noise(size);
	float sum = 0.0f;
	for (size_t i = 0; i < size; ++i) {
		noise[i] = dist(gen); // Генерация случайного значения для Дирихле
		sum += noise[i];
	}

	// Нормализация, чтобы сумма была равна 1
	for (size_t i = 0; i < size; ++i) {
		noise[i] /= sum;
	}

	return noise;
}