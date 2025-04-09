import random, os, sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import struct

import torch.utils.cpp_extension
print(torch.__version__)
torch.autograd.set_detect_anomaly(False)

ACTION_SIZE = 9
cuda = torch.cuda.is_available()

class InputLayers(nn.Module):
    def __init__(self):
        super().__init__()
        self.board_fc = nn.Linear(ACTION_SIZE * 2, 128)
        self.score_fc = nn.Linear(2, 64)
        self.player_fc = nn.Linear(1, 64)

        self.bn_board = nn.BatchNorm1d(128)
        self.bn_score = nn.BatchNorm1d(64)
        self.bn_player = nn.BatchNorm1d(64)

        self.merge_fc = nn.Linear(128 + 64 + 64, 256)
        self.bn_merge = nn.BatchNorm1d(256)

    def forward(self, x):
        if x.dim() == 1:
            # Если вход одномерный, используем простую индексацию
            board = x[:ACTION_SIZE * 2].unsqueeze(0)  # (1, ACTION_SIZE * 2)
            score = x[ACTION_SIZE * 2:ACTION_SIZE * 2 + 2].unsqueeze(0)  # (1, 2)
            player = x[ACTION_SIZE * 2 + 2].view(1, 1)  # (1, 1)
        else:
            # Если вход уже батч, используем стандартную индексацию
            board = x[:, :ACTION_SIZE * 2]
            score = x[:, ACTION_SIZE * 2:ACTION_SIZE * 2 + 2]
            player = x[:, ACTION_SIZE * 2 + 2].unsqueeze(1)

        board = F.elu(self.bn_board(self.board_fc(board)))
        score = F.relu(self.bn_score(self.score_fc(score)))
        player = torch.sigmoid(self.bn_player(self.player_fc(player)))

        x = torch.cat([board, score, player], dim=1)
        return F.silu(self.bn_merge(self.merge_fc(x)))
        
class BoardFeatures(nn.Module):
    def __init__(self, inplanes=256, planes=256):
        super().__init__()
        self.fc1 = nn.Linear(inplanes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.fc2 = nn.Linear(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)

    def forward(self, x):
        skip = x
        x = F.silu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))  # Убираем лишнюю SiLU перед сложением
        return F.silu(x + skip)

class OutputLayers(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_value1 = nn.Linear(256, 64)
        self.fc_value2 = nn.Linear(64, 1)

        self.fc_policy1 = nn.Linear(256, 128)
        self.fc_policy2 = nn.Linear(128, ACTION_SIZE)

    def forward(self, x):
        value_output = torch.tanh(self.fc_value2(F.silu(self.fc_value1(x))))
        policy_output = F.log_softmax(self.fc_policy2(F.silu(self.fc_policy1(x))), dim=1)
        return policy_output.exp(), value_output  # Можно оставить .exp() или просто softmax

class TNET(nn.Module):
    device = None
    def __init__(self):
        super(TNET, self).__init__()
        self.inputblock = InputLayers()
        for block in range(19):
            setattr(self, "res_%i" % block,BoardFeatures())
        self.outblock = OutputLayers()
    
    def forward(self,s):
        s = self.inputblock(s)
        for block in range(19):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s
    
    def save(self, path):
        with torch.no_grad():
            # Сохраняем только веса модели для дальнейшего обучения
            torch.save(self.state_dict(), path)
    
            # Генерируем пример входных данных
            _input = torch.randn(1, (2 * ACTION_SIZE) + 3)  # Размер должен соответствовать вашему forward()
    
            # Трассируем модель в TorchScript (сохраняем модель в формате для инференса)
            traced_model = torch.jit.trace(self, _input, strict=False)
    
            # Замораживаем модель (удаляем ненужные операции)
            frozen_model = torch.jit.freeze(traced_model)
    
            # Оптимизируем модель для инференса (ускоряем forward pass)
            optimized_model = torch.jit.optimize_for_inference(frozen_model)
    
            # Сохраняем оптимизированную TorchScript модель
            optimized_model.save(path.replace('.pth', '_optimized.pt'))
    
            # Сохраняем обычную TorchScript модель (для инференса без оптимизаций)
            traced_model.save(path.replace('.pth', '.pt'))
    
            print(f"Saved model weights: {path}")
            print(f"Saved TorchScript model (unoptimized): {path.replace('.pth', '.pt')}")
            print(f"Saved optimized TorchScript model: {path.replace('.pth', '_optimized.pt')}")
        
    def load(self, path, optimized=False):
        device = torch.device('cuda' if cuda else 'cpu')
        self.device = device
    
        if path.endswith('.pt'):
            try:
                if optimized:
                    self.model = torch.jit.load(path, map_location=device)
                    print("Loaded optimized TorchScript model.")
                else:
                    self.model = torch.load(path, map_location=device)
                    print("Loaded regular TorchScript model.")
            except Exception as e:
                print(f"Error loading TorchScript model: {e}")
                raise
        elif path.endswith('.pth'):
            checkpoint = torch.load(path, map_location=device)
            self.load_state_dict(checkpoint)
            print("Loaded model weights from state_dict.")
        else:
            raise ValueError("Unknown model file extension. Expected .pt or .pth")

    def save_weights(self, path):
        with open(path, 'wb') as file:
            for name, param in self.named_parameters():
                tensor = param.detach().cpu().contiguous()

                name_len = len(name)
                file.write(name_len.to_bytes(8, byteorder='little'))
                file.write(name.encode('utf-8'))

                shape = tensor.size()
                ndims = len(shape)
                file.write(ndims.to_bytes(8, byteorder='little'))
                for dim in shape:
                    file.write(dim.to_bytes(8, byteorder='little'))

                num_elems = tensor.numel()
                file.write(tensor.numpy().tobytes())

            for name, buffer in self.named_buffers():
                tensor = buffer.detach().cpu().contiguous()

                name_len = len(name)
                file.write(name_len.to_bytes(8, byteorder='little'))
                file.write(name.encode('utf-8'))

                shape = tensor.size()
                ndims = len(shape)
                file.write(ndims.to_bytes(8, byteorder='little'))
                for dim in shape:
                    file.write(dim.to_bytes(8, byteorder='little'))

                num_elems = tensor.numel()
                file.write(tensor.numpy().tobytes())

    def load_weights(self, path):
        if not os.path.exists(path):
            raise RuntimeError("Failed to open file for reading")

        with open(path, 'rb') as file:
            params_map = {name: param for name, param in self.named_parameters()}
            buffers_map = {name: buffer for name, buffer in self.named_buffers()}

            while True:
                name_len = int.from_bytes(file.read(8), byteorder='little')
                if not name_len:
                    break

                name = file.read(name_len).decode('utf-8')

                target_tensor = params_map.get(name)
                if target_tensor is None:
                    target_tensor = buffers_map.get(name)
                
                if target_tensor is None:
                    raise RuntimeError(f"Parameter or buffer '{name}' not found in model")

                ndims = int.from_bytes(file.read(8), byteorder='little')
                shape = [int.from_bytes(file.read(8), byteorder='little') for _ in range(ndims)]

                if len(shape) == 0:
                    shape = []

                expected_shape = target_tensor.shape
                
                if tuple(shape) != tuple(expected_shape):
                    raise RuntimeError(f"Shape mismatch for {name}, expected {expected_shape}, got {shape}")

                num_elems = 1
                for dim in shape:
                    num_elems *= dim
                data = file.read(num_elems * target_tensor.element_size())
                tensor_data = torch.frombuffer(data, dtype=target_tensor.dtype).reshape(shape)

                with torch.no_grad():
                    target_tensor.copy_(tensor_data)   
                    
    def act(self, state, game):
        act_values, policy = None, None
        if cuda:
            act_values, policy = self(state.cuda())
        else:
            act_values, policy = self(state)
        act_values = act_values.reshape(-1)
        possible_moves = game.getPossibleMoves()
        for i in range(ACTION_SIZE):
            if (i, game.player) not in possible_moves:
                act_values[i] = float("-inf")
        return torch.argmax(act_values)  # лучшее действие
    
class AlphaLoss(torch.nn.Module):
    """Custom loss function for AlphaZero-like training."""

    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value) ** 2
        policy_error = torch.sum((-policy * (1e-6 + y_policy).log()), dim=1)
        return (value_error.view(-1) + policy_error).mean()
    
def simulate_train(model):
    input_data = torch.randn(32, 2 * ACTION_SIZE + 3)
    policy = torch.randn(32, 9)
    value = torch.randn(32, 1)
    # Определяем критерий и оптимизатор
    criterion = AlphaLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Цикл обучения
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        policy_pred, value_pred = model(input_data)
        loss = criterion(value_pred, value, policy_pred, policy)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")
    print("Training completed.")
    
if __name__ == "__main__":
    # input_data = torch.tensor([9.000000, 9.000000, 9.000000, 9.000000, 9.000000, 9.000000, 9.000000, 9.000000, 
    #                         9.000000, 9.000000, 9.000000, 9.000000, 9.000000, 9.000000, 9.000000, 9.000000, 
    #                         9.000000, 9.000000, 0.000000, 0.000000, 0.000000], dtype=torch.float32).view(1, -1)
    # # Загрузка весов в новую модель
    # model_loaded = TNET()
    # model_loaded.eval()  # Переводим в eval перед использованием
    # model_loaded.load_weights(r"C:\Users\Akzhol\source\repos\Toguzkumalak\Toguzkumalak\build\Release\model_data\weights.dat")
    # # model_loaded.save_weights(r"C:\Users\Akzhol\source\repos\Toguzkumalak\Toguzkumalak\build\Release\model_data\py_weights.dat")
    # model_traced = torch.jit.trace(model_loaded, input_data)
    # model_freezed = torch.jit.freeze(model_traced)
    # model_optimized = torch.jit.optimize_for_inference(model_freezed)
    # torch.jit.save(model_optimized, r"C:\Users\Akzhol\source\repos\Toguzkumalak\Toguzkumalak\build\Release\model_data\best_optimized.pt")
    # # После загрузки весов
    # output_loaded = model_loaded(input_data)
    # print("After loading weights:")
    # print(output_loaded)

    torch.int16