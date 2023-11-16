import os
import glob
from tqdm import tqdm

threshold = -1.5
source_folder = 'dataset/benchmark_data_0.5'
target_folder = 'dataset/benchmark_data_0.5_pos_part'
exp_type_folers = os.listdir(source_folder)

for exp_type in tqdm(exp_type_folers):
    target_exp_type_folder = f'{target_folder}/{exp_type}'
    os.makedirs(target_exp_type_folder, exist_ok=True)
    files = glob.glob(f'{source_folder}/{exp_type}/*/*.dat')
    for file in tqdm(files):
        lines = []
        with open(file, 'r') as f:
            for line in f:
                row = line.split()
                if float(row[0]) > threshold:
                    lines.append(line)

        file_type, file_name = os.path.split(file)
        _, file_type = os.path.split(file_type)
        os.makedirs(f'{target_exp_type_folder}/{file_type}', exist_ok=True)
        target_file = f'{target_exp_type_folder}/{file_type}/{file_name}'
        with open(target_file, 'w') as f:
            f.writelines(lines)
