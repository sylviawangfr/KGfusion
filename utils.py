import json
import torch


def save2json(data:dict, out_file):
    with open(out_file, "w") as f:
        f.write(json.dumps(data))
        f.close()


def load_json(in_file):
    with open(in_file, "r") as f:
        dict_data = json.load(f)
        f.close()
    return dict_data





