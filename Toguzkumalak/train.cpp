﻿#include "pch.h"
#include "train.h"

void train(const std::shared_ptr<TNET>& model, const GameState& dataset, int thread_id, TrainConfig cfg) {
	int epochs = cfg.epochs;
	double lr = cfg.lr;
	int lr_step = cfg.lr_step;
	double gamma = cfg.gamma;
	int batch_size = cfg.batch_size;

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(lr));
    torch::optim::StepLR lr_scheduler(optimizer, /*step_size=*/lr_step, /*gamma=*/gamma);

    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    auto train_set = BoardDataset(dataset, device);
    int num_workers = std::thread::hardware_concurrency();

    auto data_loader = torch::data::make_data_loader(
        train_set.map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions()
        .batch_size(batch_size)
        .workers(num_workers)
    );

    AlphaLoss criterion;
    std::vector<float> losses_per_epoch;

    printf("[Thread %d] Training started\n", thread_id);

    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0;
        std::vector<float> losses_per_batch;
        int batch_count = 0;

        for (auto& batch : *data_loader) {
            optimizer.zero_grad(true);
            auto inputs = batch.data.to(device);
            auto policy_target = batch.target.slice(1, 0, -1).to(device);
            auto value_target = batch.target.slice(1, -1).to(device);

            auto [policy_pred, value_pred] = model->forward(inputs);

            auto loss = criterion.forward(value_pred.squeeze(), value_target, policy_pred, policy_target);

            loss.backward();
            optimizer.step();

            total_loss += loss.item<float>();
            batch_count++;

            if (batch_count % 10 == 9) {
                if (batch_count % 1000 == 9 && thread_id == 0) {
                    printf("[Thread %d] [Epoch: %d, %d/%zu points] total loss per batch: %.3f\n",
                        thread_id, epoch + 1, (batch_count + 1) * batch_size, dataset.states.size(), total_loss / 10);
                    printf("Policy: %d %d\n", policy_target[0].argmax().item<int>(), policy_pred[0].argmax().item<int>());
                    printf("Value: %.3f %.3f\n", value_target[0].item<float>(), value_pred[0].item<float>());
                }
                losses_per_batch.push_back(total_loss / 10);
                total_loss = 0.0;
            }
        }

        lr_scheduler.step();  // Обновление learning rate

        // Сохранение среднего loss за эпоху
        float avg_loss = std::accumulate(losses_per_batch.begin(), losses_per_batch.end(), 0.0f) / losses_per_batch.size();
        losses_per_epoch.push_back(avg_loss);

        // Досрочное завершение обучения, если loss стабилен
        if (losses_per_epoch.size() > 100) {
            float recent_avg = (losses_per_epoch[losses_per_epoch.size() - 2] +
                losses_per_epoch[losses_per_epoch.size() - 3] +
                losses_per_epoch[losses_per_epoch.size() - 4]) / 3;
            float past_avg = (losses_per_epoch[losses_per_epoch.size() - 14] +
                losses_per_epoch[losses_per_epoch.size() - 15] +
                losses_per_epoch[losses_per_epoch.size() - 16]) / 3;
            if (std::abs(recent_avg - past_avg) <= 0.01) {
                std::cout << "[Thread " << thread_id << "] Training early stopped at epoch " << epoch << std::endl;
                break;
            }
        }
    }
}


void start_training(const std::shared_ptr<TNET>& model, const std::string& dataset_path, int num_threads, TrainConfig cfg) {
    try {
        GameState dataset;
        printf("Loading dataset from %s\n", dataset_path.c_str());
        dataset.load(dataset_path);
        printf("Dataset loaded: %zu states, %zu policies, %zu values\n", dataset.states.size(), dataset.policies.size(), dataset.values.size());
        std::vector<std::thread> threads;
        if (num_threads > 1) {
#pragma omp parallel for
			for (int i = 0; i < num_threads; ++i) {
                train(model, dataset, i, cfg);
			}
        }
        else {
			train(model, dataset, 0, cfg);
        }
    }
    catch (const c10::Error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}