import os
import numpy as np
import yxwu_lib

from tqdm import tqdm


# if __name__ == '__main__':

#     dataloader = DataLoader(dataset)

#     y_true_list = []
#     for feat, label in tqdm(dataloader):

#         y_true_list.append(label)

#     num_pos = np.count_nonzero(x == +1)
#     num_neg = np.count_nonzero(x == -1)

#     ratio = num_pos % num_neg

#     print("number of positive 1:", num_pos)


if __name__ == '__main__':

    testCases = [
        "case_1a",
        "case_1b",
        "case_1c",
        "case_1d",
        "case_1e",
        "case_1f",
        "case_1g",
        "case_1h_1i",
        "case_1j_1n",
        "case_1o_1q",
        "case_1r_1z",
        "case_2_7"
    ]

    cwrk = os.path.abspath(os.getcwd())

    total_num_pos = 0
    total_num_neg = 0
    n = 0

    for testCase in tqdm(testCases):

        path = f"/home/haixin/workSpace/work_No1/protein-testCase/{testCase}"
        delimiter = '*.mincrd'
        proteins = yxwu_lib.get_protein(path, delimiter)
        os.chdir(cwrk)

        print(f"{testCase}")
        print("#" * 80)

        beData_path = f"/home/yxwu/install_test_mlses/yongxian-refinedMLSES/dataset/benchmark_data_0.1/{testCase}"

        proteinE_test ={}


        for protein in tqdm(proteins):

            print(f"Reading {protein} predict data...")

            bench_data = f"{beData_path}/{protein}_bench.dat"

            print(f"Classifying {protein} predict data...")
            bench = np.array(bench_data)
            bench = np.where(bench[:, 0] > 0, 1, -1)

            num_pos = np.count_nonzero(bench == +1)
            num_neg = np.count_nonzero(bench == -1)

            ratio = num_pos % num_neg

            print(f"{protein} number of +1:", num_pos)
            print(f"{protein} number of -1:", num_neg)
            print(f"{protein} +1/-1:", ratio)

            total_num_pos += num_pos
            total_num_neg += num_neg

            n += 1

    print(f"total number of +1:", total_num_pos)
    print(f"total number of -1:", total_num_neg)
    print("ratio:", total_num_pos % total_num_neg)

    print("Total number of proteins tested:", n)