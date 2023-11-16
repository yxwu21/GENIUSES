import os
import yxwu_lib
from tqdm import tqdm


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

    for testCase in tqdm(testCases):

        path = f"/home/haixin/workSpace/work_No1/protein-testCase/{testCase}"
        delimiter = '*.mincrd'
        proteins = yxwu_lib.get_protein(path, delimiter)

        print(proteins)

        folder_path = f"/home/yxwu/datasets/refinedMLSES_dataset/benchmark_data_0.2/bench_boundary_0.69/{testCase}"
        script = "Bench-boundary.run"

        os.makedirs(f"{folder_path}")
        os.system(f"cp /home/yxwu/install_test_mlses/yongxian-refinedMLSES/preprocess_data/{script} {folder_path}")

        for protein in tqdm(proteins):
            print(f"Starting calculation of proteins in {testCase} ...")
            os.chdir(f"{folder_path}")
            os.system(f"./{script} /home/haixin/workSpace/work_No1/protein-testCase/{testCase}/{protein}.p22.mincrd /home/haixin/workSpace/work_No1/protein-testCase/{testCase}/{protein}.p22.parm {protein}")
            os.rename("bench-boundary.dat", f"{protein}_bench_bnd.dat")

        print(f"Calculation of proteins in {testCase} have all completed...")
        print("*" * 160)
        os.chdir(cwrk)

    print(f"Calculation of {testCases} have completed...")