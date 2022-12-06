import os
import time
import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path


def chart_input(pos_feature, neg_feature, out_file):
    # plot the data
    fig, ax = plt.subplots()
    ax = fig.add_subplot(projection='3d')
    pos = torch.chunk(pos_feature, 3, 1)
    neg = torch.chunk(neg_feature, 3, 1)
    ax.scatter(pos[0], pos[1], pos[2], marker='o', color='tab:red')
    ax.scatter(neg[0], neg[1], neg[2], marker='^', color='tab:blue')
    # plt.show()
    plt.savefig(out_file, dpi=600)


def format_result(result):
    re = result.to_dict()
    re_str = "head hit@n: {}, {}, {}". \
        format(re['head']['realistic']['hits_at_1'],
               re['head']['realistic']['hits_at_3'],
               re['head']['realistic']['hits_at_10'])
    re_str = re_str + ';\n' + "tail hit@n: {}, {}, {}". \
        format(re['tail']['realistic']['hits_at_1'],
               re['tail']['realistic']['hits_at_3'],
               re['tail']['realistic']['hits_at_10'])
    re_str = re_str + ';\n' + "both hit@n: {}, {}, {}". \
        format(re['both']['realistic']['hits_at_1'],
               re['both']['realistic']['hits_at_3'],
               re['both']['realistic']['hits_at_10'])
    return re_str


def wait_until_file_is_saved(file_path: str, timeout_sec=10) -> bool:
    time_counter = 0
    interval = int(timeout_sec / 10) if timeout_sec > 10 else 1
    print(f"waiting for saving {file_path} ...")
    while not os.path.exists(file_path):
        time.sleep(interval)
        time_counter += interval
        # print(f"waiting {time_counter} sec.")
        if time_counter > timeout_sec:
            # print(f"saving {file_path} timeout")
            break
    is_saved = os.path.exists(file_path)
    if is_saved:
        print(f"{file_path} has been saved.")
    else:
        print(f"saving {file_path} timeout")
    return is_saved


def save_to_file(text, out_filename, mode='w'):
    out_path = Path(out_filename)
    if not out_path.parent.exists():
        out_path.parent.mkdir(exist_ok=False)
    with open(out_path, encoding='utf-8', mode=mode) as out_f:
        out_f.write(text)


def init_dir(work_dir):
    out_path = Path(work_dir)
    if not out_path.exists():
        out_path.mkdir(exist_ok=False)
