import os
import re
import argparse
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='./data/steg',
                        help='Steg data directory')
    parser.add_help = True
    args = parser.parse_args()
    steg_data_dir = args.d

    steg_path = './K_3/data_K_3/data_processed/steg.txt'

    total_length = 0 

    for i in range(3, 4):
        print(i)
        dna_sequences = []
        filenames = [filename for filename in os.listdir(steg_data_dir) if filename.startswith(f'adg_fxy{i}')]
        print('i ',i)
        print('filenames  ',filenames)
        pattern = re.compile(r"\[[ATCG', ]+]")
        for filename in filenames:
            with open(os.path.join(steg_data_dir, filename), 'r', encoding='gb2312') as f:
                dna_sequences += [''.join(eval(s)) for s in pattern.findall(f.read())]
        

        total_length +=len(dna_sequences)


        with open(steg_path , 'a') as f:
            for k in [3]:
                for s in range(1): 
                    print(f'(Steg) k = {k}, s = {s}:')
                    for seq in tqdm(dna_sequences, desc="Processing", unit="seq"):
                        for i in range(s, len(seq), k):
                            if i + k <= len(seq):
                                f.write(seq[i:i+k] + ' ')
                        f.write('\n')


    print("Total length of dna_sequences:", total_length)


  