import os
import random

def copy_random_lines(source_file, destination_file, train_and_val_file, percentage):
    with open(source_file, 'r') as f:
        lines = f.readlines()


    num_lines = len(lines)
    num_lines_to_copy = int(num_lines * percentage / 100)


    selected_indices = random.sample(range(num_lines), num_lines_to_copy)


    with open(destination_file, 'w') as f_dest, open(train_and_val_file, 'w') as f_train_val:
        for idx, line in enumerate(lines):
            if idx in selected_indices:
                f_dest.write(line) 
            else:
                f_train_val.write(line) 

raw_source_file = "./K_3/data_K_3/data_processed/raw_del.txt"
steg_source_file = "./K_3/data_K_3/data_processed/steg_del.txt"


raw_destination_val_file = "./K_3/data_K_3/test/raw.txt"
raw_destination_train_file = "./K_3/data_K_3/train_val/raw.txt"



steg_destinatione_val_file = "./K_3/data_K_3/test/steg.txt"
steg_destinatione_train_file = "./K_3/data_K_3/train_val/steg.txt"


copy_random_lines(raw_source_file, raw_destination_val_file,raw_destination_train_file, 10)

copy_random_lines(steg_source_file, steg_destinatione_val_file,steg_destinatione_train_file, 10)
