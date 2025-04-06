#pragma once
#include "node.h"
#include "tnet.h"


#ifdef _WIN32
#define NOMINMAX  // ����� �������� ���������� � `min` � `max` �� Windows.h
#include <Windows.h>
#else
#include <unistd.h>
#include <fstream>
#endif

struct GameState {
    std::vector<std::vector<float>> states;
    std::vector<std::vector<float>> policies;
    std::vector<float> values;

    void save(const std::string& filename) const {
        std::ofstream out(filename, std::ios::binary);

        // ��������� ���������� ��������� � ������ �������
        size_t states_size = states.size();
        size_t policies_size = policies.size();
        size_t values_size = values.size();

        out.write(reinterpret_cast<const char*>(&states_size), sizeof(states_size));
        out.write(reinterpret_cast<const char*>(&policies_size), sizeof(policies_size));
        out.write(reinterpret_cast<const char*>(&values_size), sizeof(values_size));

        // ��������� ������ ��������
        for (const auto& state : states) {
            size_t state_size = state.size();
            out.write(reinterpret_cast<const char*>(&state_size), sizeof(state_size));
            out.write(reinterpret_cast<const char*>(state.data()), state_size * sizeof(float));
        }

        for (const auto& policy : policies) {
            size_t policy_size = policy.size();
            out.write(reinterpret_cast<const char*>(&policy_size), sizeof(policy_size));
            out.write(reinterpret_cast<const char*>(policy.data()), policy_size * sizeof(float));
        }

        out.write(reinterpret_cast<const char*>(values.data()), values_size * sizeof(float));
        out.close();
    }

    void load(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Failed to open dataset file");
        }

        size_t states_size, policies_size, values_size;
        in.read(reinterpret_cast<char*>(&states_size), sizeof(states_size));
        in.read(reinterpret_cast<char*>(&policies_size), sizeof(policies_size));
        in.read(reinterpret_cast<char*>(&values_size), sizeof(values_size));

        states.resize(states_size);
        policies.resize(policies_size);
        values.resize(values_size);

        for (auto& state : states) {
            size_t state_size;
            in.read(reinterpret_cast<char*>(&state_size), sizeof(state_size));
            state.resize(state_size);
            in.read(reinterpret_cast<char*>(state.data()), state_size * sizeof(float));
        }

        for (auto& policy : policies) {
            size_t policy_size;
            in.read(reinterpret_cast<char*>(&policy_size), sizeof(policy_size));
            policy.resize(policy_size);
            in.read(reinterpret_cast<char*>(policy.data()), policy_size * sizeof(float));
        }

        in.read(reinterpret_cast<char*>(values.data()), values_size * sizeof(float));
        in.close();
    }
};

std::string current_time();
std::string elapsed_time(long long seconds);
std::string elapsed_time(float seconds);
std::string current_date();

torch::jit::script::Module load_model(const std::string& model_path);
std::pair<std::vector<float>, float> net_func(torch::jit::script::Module model, Game* game);
std::vector<float> softmax(const std::vector<float>& x);
std::vector<float> get_policy(UCTNode* root, float temperature = 1.0);
std::pair<int, std::vector<float>> UCT_search(torch::jit::script::Module model, Game* game, int num_reads, bool selfplay);
void MCTS_self_play(std::string model_path, std::string save_path, int num_games = 25, int cpu = 0, bool affinity = true);
void self_play(std::string model_path, int num_games = 1, int depth = 2, int ai_side = 0);
void play_against_alphazero(std::string model_path, int ai_side = 0);

