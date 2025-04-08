import torch
import struct
from brain import *

def compare_weights_directly(model, path_to_check):
    with open(path_to_check, "rb") as file:
        index = 0
        for param in model.parameters():
            # Читаем размер параметра
            size = struct.unpack('<Q', file.read(8))[0]
            param_flat = param.data.view(-1)
            if size != param_flat.numel():
                raise RuntimeError(f"Size mismatch: expected {param_flat.numel()}, got {size}")
            # Читаем данные
            param_data = struct.unpack(f'<{size}f', file.read(size * 4))
            for i, value in enumerate(param_data):
                if abs(param_flat[i].item() - value) > 1e-5:  # Проверка с маленьким допуском
                    print(f"Mismatch at parameter #{index}, element {i}: {param_flat[i].item()} vs {value}")
                    return
            index += 1
        print("✅ All weights match.")

# Пример вызова:
model = TNET()
model.eval()
model.load_binary_weights(r"C:\Users\Akzhol\source\repos\Toguzkumalak\Toguzkumalak\build\Release\model_data\weights.dat")
compare_weights_directly(model, r"C:\Users\Akzhol\source\repos\Toguzkumalak\Toguzkumalak\build\Release\model_data\py_weights.dat")