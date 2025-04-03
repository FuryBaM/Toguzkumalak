#pragma once
#include "mcts_selfplay.h"

const bool cuda_available = torch::cuda::is_available();

struct BoardDataset : torch::data::datasets::Dataset<BoardDataset> {
    std::vector<torch::Tensor> states, policies, values;

    // Конструктор, аналогичный Python
    explicit BoardDataset(const GameState& dataset, torch::Device device) {
        for (size_t i = 0; i < dataset.states.size(); i++) {
            // Преобразуем состояния, политики и значения в тензоры и отправляем на нужное устройство
            states.push_back(torch::tensor(dataset.states[i], torch::dtype(torch::kFloat32)).to(device));
            policies.push_back(torch::tensor(dataset.policies[i], torch::dtype(torch::kFloat32)).to(device));
            values.push_back(torch::tensor({ dataset.values[i] }, torch::dtype(torch::kFloat32)).to(device));
        }
    }

    torch::data::Example<> get(size_t index) override {
        // Объединяем политику и значение в один тензор
        auto target = torch::cat({ policies[index], values[index] }, 0);
        return { states[index], target };
    }

    // Размер датасета
    torch::optional<size_t> size() const override {
        return states.size();
    }
};


struct AlphaLoss : torch::nn::Module {
    torch::Tensor forward(torch::Tensor y_value, torch::Tensor value, torch::Tensor y_policy, torch::Tensor policy) {
        value.set_requires_grad(true);
        policy.set_requires_grad(true);

        torch::Tensor value_error = torch::pow(value - y_value, 2);
        torch::Tensor policy_error = torch::sum(-policy * (y_policy + 1e-6).log(), 1);

        torch::Tensor loss = torch::mean(value_error + policy_error);

        return loss;
    }
};

void train(const std::shared_ptr<TNET>& model, const GameState& dataset, int epoch_start, int epoch_stop, int thread_id);
void start_training(const std::shared_ptr<TNET>& model, const std::string& dataset_path, int epochs, int num_threads);


