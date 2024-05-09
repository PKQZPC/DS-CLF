k = 3  # 设置 k 的值，即几个碱基为一组

# 映射表的初始值
mapping = {}

# 根据 k 的值生成映射表
bases = ["A", "C", "G", "T"]
count = 3
for i in range(len(bases)**k):
    key = ''.join([bases[(i // (len(bases)**j)) % len(bases)] for j in range(k)])
    mapping[key] = count
    count += 1

# 将映射表写入文件
with open("./K_3/DNA_map_k_3.txt", "w") as f:
    for key, value in mapping.items():
        f.write(f"{key}\t{value}\n")
