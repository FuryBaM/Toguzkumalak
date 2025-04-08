import random, os, sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import struct
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
    
    # void save_weights(const std::string& path) {
    #     std::ofstream file(path, std::ios::binary);
    #     to(torch::kCPU);
    #     for (const auto& param : parameters()) {
    #         auto data = param.data().contiguous();
    #         unsigned long long size = data.numel();
    #         file.write(reinterpret_cast<const char*>(&size), sizeof(size));
    #         file.write(reinterpret_cast<const char*>(data.data_ptr<float>()), size * sizeof(float));
    #     }
    #     file.close();
    # }
    
    def save_binary_weights(self, path):
        self.to(torch.device('cpu'))
        with open(path, "wb") as file:
            for name, param in self.named_parameters():  # Используем named_parameters
                data = param.data.contiguous().cpu().view(-1).numpy().astype(np.float32)
                size = len(data)
                # Записываем имя параметра (для проверки, что загружаем тот же параметр)
                name_len = len(name)
                file.write(struct.pack('<q', name_len))  # Длина имени параметра
                file.write(name.encode('utf-8'))  # Сохраняем имя параметра в байтовом виде
                # Записываем размер и данные параметра
                file.write(struct.pack('<q', size))  # Длина данных параметра
                file.write(data.tobytes())  # Запись данных параметра

    def load_binary_weights(self, path: str):
        self.to(torch.device('cpu'))
        with open(path, "rb") as file:
            for name, param in self.named_parameters():
                # Чтение имени параметра
                name_len = struct.unpack('<q', file.read(8))[0]
                name_read = file.read(name_len).decode('utf-8')
                
                if name_read != name:
                    print(f"Warning: Mismatch in parameter name. Expected '{name}', but found '{name_read}'")
                    continue  # Если имя параметра не совпадает, пропускаем его

                # Чтение размера и данных параметра
                size = struct.unpack('<q', file.read(8))[0]
                param_data = np.frombuffer(file.read(size * 4), dtype=np.float32)
                param.data.copy_(torch.tensor(param_data).view_as(param))
    
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
    input_data = torch.randn(32, 2*ACTION_SIZE+3)
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
    model = TNET()
    model.eval()
    
    # Генерация входных данных
    input_data = torch.tensor([9.000000, 9.000000, 9.000000, 9.000000, 9.000000, 9.000000, 9.000000, 9.000000, 
                            9.000000, 9.000000, 9.000000, 9.000000, 9.000000, 9.000000, 9.000000, 9.000000, 
                            9.000000, 9.000000, 0.000000, 0.000000, 0.000000], dtype=torch.float32).view(1, -1)
    
    # Перед тренировкой
    output = model(input_data)
    print("Before training:")
    print(output)

    # Обучение модели
    simulate_train(model)
    input_data = torch.tensor([9.000000, 9.000000, 9.000000, 9.000000, 9.000000, 9.000000, 9.000000, 9.000000, 
                            9.000000, 9.000000, 9.000000, 9.000000, 9.000000, 9.000000, 9.000000, 9.000000, 
                            9.000000, 9.000000, 0.000000, 0.000000, 0.000000], dtype=torch.float32).view(1, -1)
    # После тренировки
    input_data_batch = input_data.repeat(2, 1)
    output = model(input_data_batch)
    print("Before save:")
    print(output)
    model.eval()
    # Сохранение весов
    #model.save_binary_weights(r"C:\Users\Akzhol\source\repos\Toguzkumalak\Toguzkumalak\build\Release\model_data\py_weights.dat")
    input_data = torch.tensor([9.000000, 9.000000, 9.000000, 9.000000, 9.000000, 9.000000, 9.000000, 9.000000, 
                            9.000000, 9.000000, 9.000000, 9.000000, 9.000000, 9.000000, 9.000000, 9.000000, 
                            9.000000, 9.000000, 0.000000, 0.000000, 0.000000], dtype=torch.float32).view(1, -1)
    # Загрузка весов в новую модель
    model_loaded = TNET()
    model_loaded.load_binary_weights(r"C:\Users\Akzhol\source\repos\Toguzkumalak\Toguzkumalak\build\Release\model_data\weights.dat")
    model_loaded.eval()  # Переводим в eval перед использованием

    # После загрузки весов
    output_loaded = model_loaded(input_data)
    print("After loading weights:")
    print(output_loaded)

    # Печать параметров модели (если нужно для диагностики)
    # print("Model parameters after loading weights:")
    # for name, param in model_loaded.named_parameters():
    #     print(f"{name}: {param}")
