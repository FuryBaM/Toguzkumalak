#pragma once

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
    torch::nn::ModuleHolder<InputLayers> inputblock;
    std::vector<torch::nn::ModuleHolder<BoardFeatures>> res_blocks;
    torch::nn::ModuleHolder<OutputLayers> outblock;

    TNET() {
        inputblock = register_module("inputblock", std::make_shared<InputLayers>());

        for (int i = 0; i < 9; ++i) {
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
};