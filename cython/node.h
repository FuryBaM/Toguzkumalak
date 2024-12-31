#pragma once
#include "game.h"
#include <unordered_map>
#include <random>
class UCTNode
{
public:
	Game* game;
	size_t action_size;
	int move;
	bool is_expanded;
	bool self_play;
	UCTNode* parent;
	std::unordered_map<int, UCTNode*> children;
	std::vector<float> child_priors;
	std::vector<float> child_total_value;
	std::vector<float> child_number_visits;
	std::vector<int> action_idxes;
	float a;

	UCTNode(Game* game, int move = -1, UCTNode* parent = nullptr, bool selfplay = false);
	void DestroyAllChildren();
	~UCTNode();
	float getNumberVisits()
	{
		if (move == -1) return 0.0f;
		return parent->child_number_visits[move];
	}
	void setNumberVisits(float value)
	{
		if (move == -1) return;
		parent->child_number_visits[move] = value;
	}
	float getTotalValue()
	{
		if (move == -1) return 0.0f;
		return parent->child_total_value[move];
	}
	void setTotalValue(float value)
	{
		if (move == -1) return;
		parent->child_total_value[move] = value;
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
template <typename T>
std::vector<T> vectorDivide(std::vector<T> first, std::vector<T> second)
{
	size_t size1 = first.size();
	size_t size2 = second.size();
	if (size1 != size2) {
		throw std::invalid_argument("Vectors must have the same size for division.");
	}
	std::vector<T> result(size1);
	for (size_t i = 0; i < size1; ++i)
	{
		if (second[i] == 0)
		{
			result[i] = std::numeric_limits<T>::infinity();
			continue;
		}
		result[i] = first[i] / second[i];
	}
	return result;
}

template <typename T>
std::vector<T> vectorMultiply(std::vector<T> first, std::vector<T> second)
{
	size_t size1 = first.size();
	size_t size2 = second.size();
	if (size1 != size2) {
		throw std::invalid_argument("Vectors must have the same size for multiplication.");
	}
	std::vector<T> result(size1);
	for (size_t i = 0; i < size1; ++i)
	{
		result[i] = first[i] * second[i];
	}
	return result;
}

template <typename T>
std::vector<T> vectorAdd(std::vector<T> first, std::vector<T> second)
{
	size_t size1 = first.size();
	size_t size2 = second.size();
	if (size1 != size2) {
		throw std::invalid_argument("Vectors must have the same size for addition.");
	}
	std::vector<T> result(size1);
	for (size_t i = 0; i < size1; ++i)
	{
		result[i] = first[i] + second[i];
	}
	return result;
}

template <typename T>
std::vector<T> absVector(std::vector<T> v)
{
	size_t size = v.size();
	for (size_t i = 0; i < size; ++i)
	{
		v[i] = abs(v[i]);
	}
	return v;
}

template <typename T>
std::vector<T> addNumberToVector(const std::vector<T>& v, T number)
{
	std::vector<T> result = v;
	size_t size = result.size();
	for (size_t i = 0; i < size; ++i)
	{
		result[i] += number;  // добавляем число ко всем элементам
	}
	return result;  // возвращаем измененный вектор
}

template <typename T>
std::vector<T> multiplyNumberToVector(const std::vector<T>& v, T number)
{
	std::vector<T> result = v;
	size_t size = result.size();
	for (size_t i = 0; i < size; ++i)
	{
		result[i] *= number;
	}
	return result;
}

template<typename T>
bool contains(const std::vector<T>& vec, const T& value) {
	return std::find(vec.begin(), vec.end(), value) != vec.end();
}
template<typename K, typename V>
bool contains(const std::unordered_map<K, V>& map, const K& key) {
	return map.find(key) != map.end();
}

int argmax(const std::vector<float>& vec);
int argmax(std::vector<float> vec, std::vector<int> indices);
std::vector<float> generate_dirichlet_noise(size_t size);