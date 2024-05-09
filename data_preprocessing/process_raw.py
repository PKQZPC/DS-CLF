import os
import argparse
from tqdm import tqdm


k=3
start=0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=k, help='Segmentation size')
    parser.add_argument('-s', type=int, default=start, help='Start index')
    parser.add_argument('-o', type=str, default=f'./K_3/data_K_3/data_processed/raw.txt', help="Output file's path")
    parser.add_argument('-d', type=str, default='./data/raw', help='Raw data directory')
    parser.add_help = True
    args = parser.parse_args()
    k = args.k
    start = args.s
    output_path = args.o
    raw_data_dir = args.d
    dna_sequences = []
    for filename in os.listdir(raw_data_dir):
        with open(os.path.join(raw_data_dir, filename), 'r') as f:
            dna_sequences += [line for line in f.read().splitlines() if line != '']
    with open(output_path, 'w') as f:
        for dna_sequence in tqdm(dna_sequences, desc="Processing", unit="seq"):
            for i in range(start, len(dna_sequence), k):
                if i + k <= len(dna_sequence):
                    f.write(dna_sequence[i:i+k] + ' ')
            f.write('\n')


    # 读取已经生成的 steg.txt 文件
    with open(output_path , 'r') as f:
        lines = f.readlines()

    # 计算最小长度
    min_length = min(len(line.split()) for line in lines)

    # 将所有行调整为相同的长度
    fixed_lines = [line.split()[:min_length] for line in lines]

    # 写入修正后的内容到文件中
    with open(output_path , 'w') as f:
        for line in fixed_lines:
            f.write(' '.join(line) + '\n')