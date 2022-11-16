import matplotlib.pyplot as plt
import torch
import numpy as np


def chart_rel_eval(in_file):
    ht_rel_eval = torch.load(in_file)
    # plot the data
    fig, ax = plt.subplots()
    h = torch.sort(ht_rel_eval[:, 0])
    t = torch.sort(ht_rel_eval[:, 1])
    ax.plot(np.arange(1, ht_rel_eval.shape[0] + 1), h.values, color='tab:blue')
    ax.plot(np.arange(1, ht_rel_eval.shape[0] + 1), t.values, color='tab:red')
    print("test")

if __name__ == '__main__':
    chart_rel_eval("outputs/rel_eval.pt")