import matplotlib.pyplot as plt
import torch
import numpy as np


def chart_rel_eval(in_file):
    ht_rel_eval = torch.load(in_file)
    # plot the data
    fig, ax = plt.subplots()
    h = torch.sort(ht_rel_eval[:, 0])
    h_index = h.indices
    h_value = h.values
    t_value = torch.index_select(ht_rel_eval[:, 1], 0, h_index)
    # h_value = ht_rel_eval[:, 0]
    # t_value = ht_rel_eval[:, 1]
    ax.plot(np.arange(1, ht_rel_eval.shape[0] + 1), h_value, color='tab:blue')
    ax.plot(np.arange(1, ht_rel_eval.shape[0] + 1), t_value, color='tab:red')
    print("test")

if __name__ == '__main__':
    chart_rel_eval("outputs/rel_eval.pt")