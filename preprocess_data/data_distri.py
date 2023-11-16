import matplotlib.pyplot as plt
import numpy as np

from matplotlib import colors
from matplotlib.ticker import PercentFormatter


if __name__ == '__main__':

    dat_bound_path = "/home/yxwu/install_test_mlses/yongxian-refinedMLSES/dataset/benchmark_data_0.2/bench_boundary/*/*.dat"
    dat_pos_path = "/home/yxwu/install_test_mlses/yongxian-refinedMLSES/dataset/benchmark_data_0.2/bench_positive/*/*.dat"
    dat_neg_path = "/home/yxwu/install_test_mlses/yongxian-refinedMLSES/dataset/benchmark_data_0.2/bench_negative/*/*.dat"

    bound_dat =
    bound_pos =
    bound_neg =

    dis_bound = np.array(bound_dat[:, 0])
    dis_pos = np.array(pos_dat[:, 0])
    dis_neg = np.array(bound_neg[:, 0])

    fig_bound, axs_bound = plt.subplots(1, 1, sharey=True, tight_layout=True)
    axs_bound.hist(dis_bound)
    fig_bound.savefig("boundary distribution")

    fig_pos, axs_pos = plt.subplots(1, 1, sharey=True, tight_layout=True)
    axs_pos.hist(dis_pos)
    fig_pos.savefig("positive distribution")

    fig_neg, axs_neg = plt.subplots(1, 1, sharey=True, tight_layout=True)
    axs_neg.hist(dis_neg)
    fig_neg.savefig("negative distribution")