﻿#pragma once

constexpr int ACTION_SIZE = 9;

class InputLayers : public torch::nn::Module {
public:
    torch::nn::Linear board_fc{ nullptr };
    torch::nn::Linear score_fc{ nullptr };
    torch::nn::Linear player_fc{ nullptr };
    torch::nn::BatchNorm1d bn_board{ nullptr };
    torch::nn::BatchNorm1d bn_score{ nullptr };
    torch::nn::BatchNorm1d bn_player{ nullptr };
    torch::nn::Linear merge_fc{ nullptr };
    torch::nn::BatchNorm1d bn_merge{ nullptr };

    InputLayers() {
        board_fc = register_module("board_fc", torch::nn::Linear(ACTION_SIZE * 2, 128));
        score_fc = register_module("score_fc", torch::nn::Linear(2, 64));
        player_fc = register_module("player_fc", torch::nn::Linear(1, 64));

        bn_board = register_module("bn_board", torch::nn::BatchNorm1d(128));
        bn_score = register_module("bn_score", torch::nn::BatchNorm1d(64));
        bn_player = register_module("bn_player", torch::nn::BatchNorm1d(64));

        merge_fc = register_module("merge_fc", torch::nn::Linear(128 + 64 + 64, 256));
        bn_merge = register_module("bn_merge", torch::nn::BatchNorm1d(256));
    }

    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor board, score, player;

        if (x.dim() == 1) {
            board = x.narrow(0, 0, ACTION_SIZE * 2).unsqueeze(0);
            score = x.narrow(0, ACTION_SIZE * 2, 2).unsqueeze(0);
            player = x.narrow(0, ACTION_SIZE * 2 + 2, 1).unsqueeze(0);
        }
        else {
            board = x.narrow(1, 0, ACTION_SIZE * 2);
            score = x.narrow(1, ACTION_SIZE * 2, 2);
            player = x.narrow(1, ACTION_SIZE * 2 + 2, 1);
        }

        board = torch::elu(bn_board(board_fc(board)));
        score = torch::relu(bn_score(score_fc(score)));
        player = torch::sigmoid(bn_player(player_fc(player)));

        auto x_cat = torch::cat({ board, score, player }, 1);
        return torch::silu(bn_merge(merge_fc(x_cat)));
    }
};


class BoardFeatures : public torch::nn::Module {
public:
    torch::nn::Linear fc1{ nullptr };
    torch::nn::BatchNorm1d bn1{ nullptr };
    torch::nn::Linear fc2{ nullptr };
    torch::nn::BatchNorm1d bn2{ nullptr };

    BoardFeatures(int inplanes = 256, int planes = 256) {
        fc1 = register_module("fc1", torch::nn::Linear(inplanes, planes));
        bn1 = register_module("bn1", torch::nn::BatchNorm1d(planes));
        fc2 = register_module("fc2", torch::nn::Linear(planes, planes));
        bn2 = register_module("bn2", torch::nn::BatchNorm1d(planes));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto skip = x;
        x = torch::silu(bn1(fc1(x)));
        x = bn2(fc2(x));  // Without SiLU before the addition
        return torch::silu(x + skip);
    }
};


class OutputLayers : public torch::nn::Module {
public:
    torch::nn::Linear fc_value1{ nullptr };
    torch::nn::Linear fc_value2{ nullptr };

    torch::nn::Linear fc_policy1{ nullptr };
    torch::nn::Linear fc_policy2{ nullptr };

    OutputLayers() {
        fc_value1 = register_module("fc_value1", torch::nn::Linear(256, 64));
        fc_value2 = register_module("fc_value2", torch::nn::Linear(64, 1));

        fc_policy1 = register_module("fc_policy1", torch::nn::Linear(256, 128));
        fc_policy2 = register_module("fc_policy2", torch::nn::Linear(128, ACTION_SIZE));
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        auto value_output = torch::tanh(fc_value2(torch::silu(fc_value1(x))));
        auto policy_output = torch::log_softmax(fc_policy2(torch::silu(fc_policy1(x))), 1);
        return std::make_tuple(policy_output.exp(), value_output); // Can use exp() or softmax
    }
};


class TNET : public torch::nn::Module {
public:
    std::shared_ptr<InputLayers> inputblock;
    std::vector<std::shared_ptr<BoardFeatures>> res_blocks;
    std::shared_ptr<OutputLayers> outblock;

    TNET() {
        inputblock = register_module("inputblock", std::make_shared<InputLayers>());

        for (int i = 0; i < 19; ++i) {
            res_blocks.push_back(register_module("res_" + std::to_string(i), std::make_shared<BoardFeatures>()));
        }

        outblock = register_module("outblock", std::make_shared<OutputLayers>());
    }


    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        x = inputblock->forward(x);
        for (auto& block : res_blocks) {
            x = block->forward(x);
        }
        return outblock->forward(x);
    }

    void save_weights(const std::string& path) {
        std::ofstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error("Failed to open file for writing");

        for (const auto& pair : named_parameters()) {
            const std::string& name = pair.key();
            const torch::Tensor& param = pair.value();
            torch::Tensor tensor = param.detach().cpu().contiguous();

            int64_t name_len = name.size();
            file.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
            file.write(name.data(), name_len);
            c10::ScalarType type = tensor.scalar_type();
            auto shape = tensor.sizes();
            int64_t ndims = shape.size();
            file.write(reinterpret_cast<const char*>(&ndims), sizeof(ndims));
            for (auto dim : shape) {
                file.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
            }

            int64_t num_elems = tensor.numel();
            file.write(reinterpret_cast<const char*>(tensor.data_ptr()), num_elems * tensor.element_size());
        }

        for (const auto& pair : named_buffers()) {
            const std::string& name = pair.key();
            const torch::Tensor& param = pair.value();
            torch::Tensor tensor = param.detach().cpu().contiguous();

            int64_t name_len = name.size();
            file.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
            file.write(name.data(), name_len);

            auto shape = tensor.sizes();
            int64_t ndims = shape.size();
            file.write(reinterpret_cast<const char*>(&ndims), sizeof(ndims));
            for (auto dim : shape) {
                file.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
            }

            int64_t num_elems = tensor.numel();
            file.write(reinterpret_cast<const char*>(tensor.data_ptr()), num_elems * tensor.element_size());
        }
    }
    void load_weights(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error("Failed to open file for reading");

        std::unordered_map<std::string, torch::Tensor> params_map;
        for (auto& pair : named_parameters()) {
            params_map[pair.key()] = pair.value();
        }

        std::unordered_map<std::string, torch::Tensor> buffers_map;
        for (auto& pair : named_buffers()) {
            buffers_map[pair.key()] = pair.value();
        }

        while (file.peek() != EOF) {
            int64_t name_len;
            file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
            std::string name(name_len, '\0');
            file.read(name.data(), name_len);

            torch::Tensor* target_tensor = nullptr;
            auto it_param = params_map.find(name);
            if (it_param != params_map.end()) {
                target_tensor = &it_param->second;
            }
            else {
                auto it_buf = buffers_map.find(name);
                if (it_buf != buffers_map.end()) {
                    target_tensor = &it_buf->second;
                }
                else {
                    throw std::runtime_error("Parameter or buffer " + name + " not found in model");
                }
            }

            torch::Tensor& param = *target_tensor;

            int64_t ndims;
            file.read(reinterpret_cast<char*>(&ndims), sizeof(ndims));
            std::vector<int64_t> shape(ndims);
            for (int i = 0; i < ndims; ++i) {
                file.read(reinterpret_cast<char*>(&shape[i]), sizeof(shape[i]));
            }

            auto expected_shape = param.sizes();
            if (shape != expected_shape) {
                throw std::runtime_error("Shape mismatch for " + name);
            }

            int64_t num_elems = 1;
            for (auto dim : shape) num_elems *= dim;
            file.read(reinterpret_cast<char*>(param.data_ptr()), num_elems * param.element_size());
        }
    }
};