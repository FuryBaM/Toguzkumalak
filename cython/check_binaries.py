import struct

def compare_weight_files(path1, path2):
    with open(path1, "rb") as f1, open(path2, "rb") as f2:
        index = 0
        while True:
            # Читаем 8 байт для размера
            size_bytes_1 = f1.read(8)
            size_bytes_2 = f2.read(8)

            # Если достигнут конец одного из файлов
            if not size_bytes_1 or not size_bytes_2:
                print("✅ Done comparing.")
                break

            # Распаковываем размер как long long (8 байт)
            size_1 = struct.unpack("Q", size_bytes_1)[0]
            size_2 = struct.unpack("Q", size_bytes_2)[0]

            if size_1 != size_2:
                print(f"❌ Mismatch in tensor size at param #{index}: {size_1} vs {size_2}")
                return

            # Читаем и распаковываем данные для каждого файла
            data_1 = struct.unpack(f"{size_1}f", f1.read(size_1 * 4))
            data_2 = struct.unpack(f"{size_2}f", f2.read(size_2 * 4))

            # Сравниваем данные
            for i, (a, b) in enumerate(zip(data_1, data_2)):
                if abs(a - b) > 1e-5:
                    print(f"❌ Mismatch at param #{index}, element {i}: {a} vs {b}")
                    return
            index += 1
        print("✅ All weights match.")
        
if __name__ == "__main__":
    path1 = "C:/Users/Akzhol/source/repos/Toguzkumalak/Toguzkumalak/build/Release/model_data/weights.dat"
    path2 = "C:/Users/Akzhol/source/repos/Toguzkumalak/Toguzkumalak/build/Release/model_data/py_weights.dat"
    compare_weight_files(path1, path2)