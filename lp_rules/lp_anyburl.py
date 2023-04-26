import argparse
import os
import os.path as osp
from itertools import zip_longest
import pykeen.datasets
import torch
import pandas as pd
from pykeen.datasets import get_dataset

from analysis.group_eval_utils import get_all_pos_triples, to_fusion_eval_format
from common_utils import save_to_file, wait_until_file_is_saved, init_dir


def read_hrt_pred_anyburl(anyburl_dir, snapshot=100, top_k=10):
    with open(anyburl_dir + f"predictions/alpha-{snapshot}") as f:
        lines = f.readlines()
        chunks = zip_longest(*[iter(lines)] * 3, fillvalue='')
        h_preds = []
        t_preds = []
        file_triples = []
        for chunk in chunks:
            h, r, t = chunk[0].strip().split()
            file_triples.append([int(h), int(r), int(t)])
            hs = chunk[1][7:].strip().split('\t')
            h_preds.append(torch.as_tensor([float(hs[i]) if (i < len(hs) and hs[i] != '') else -1 for i in range(top_k * 2)]))
            ts = chunk[2][7:].strip().split('\t')
            t_preds.append(torch.as_tensor([float(ts[i]) if (i < len(ts) and ts[i] != '') else -1 for i in range(top_k * 2)]))
    h_preds = torch.stack(h_preds, 0)
    t_preds = torch.stack(t_preds, 0)
    preserve_shape = h_preds.shape
    h_preds = h_preds.reshape([preserve_shape[0] * top_k, 2])
    h_index = h_preds[:, 0].reshape([preserve_shape[0], top_k]).type(torch.int64)
    h_scores = h_preds[:, 1].reshape([preserve_shape[0], top_k])
    t_preds = t_preds.reshape([preserve_shape[0] * top_k, 2])
    t_index = t_preds[:, 0].reshape([preserve_shape[0], top_k]).type(torch.int64)
    t_scores = t_preds[:, 1].reshape([preserve_shape[0], top_k])
    pred_index = torch.stack([h_index, t_index], 1)
    pred_scores = torch.stack([h_scores, t_scores], 1)
    return torch.as_tensor(file_triples, dtype=torch.int64), pred_index, pred_scores


def anyburl_to_pykeen_format(mapped_triples, num_candidates, anyburl_dir, snapshot=100, top_k=10):
    # [num_tri, h_scores + t_scores]
    file_triples, pred_index, pred_scores = read_hrt_pred_anyburl(anyburl_dir, snapshot, top_k)
    pred_index[pred_index == -1] = num_candidates  # add one extra column to handle mask value/index
    pykeen_scatter_scores = torch.zeros([pred_scores.shape[0], 2, num_candidates + 1]).scatter_(2, pred_index, pred_scores) # sigmoid scores [0-1]
    # get index of triples
    file_index = get_index_of_a_in_b(file_triples, mapped_triples)
    converted = torch.zeros([mapped_triples.shape[0], 2, num_candidates])
    converted[file_index] = pykeen_scatter_scores[:, :, :-1]
    ht = torch.chunk(converted, 2, 1)
    ht_converted = torch.cat([ht[0].squeeze(1), ht[1].squeeze(1)], 1)
    return ht_converted


# def per_rel_eval(mapped_triples, pred_scores, out_dir):
#     original_df = pd.DataFrame(data=mapped_triples.numpy(), columns=['h', 'r', 't'])
#     original_groups = original_df.groupby('r', group_keys=True, as_index=False)
#     ordered_keys = original_groups.groups.keys()
#     head_tail_mrr = []
#     for rel in ordered_keys:
#         rg_tris = original_groups.get_group(rel)
#         rg_index = torch.as_tensor(rg_tris.index)
#         h_preds, t_preds = pred_scores[rg_index].chunk(2, 1)
#         heads = torch.as_tensor(rg_tris['h'].values)
#         tails = torch.as_tensor(rg_tris['t'].values)
#         head_hits = calc_hit_at_k(h_preds, heads)
#         tail_hits = calc_hit_at_k(t_preds, tails)
#         both_hit = calc_hit_at_k(torch.cat([h_preds, t_preds]), torch.cat((heads, tails)))
#         head_tail_mrr.append(torch.as_tensor([head_hits,
#                               tail_hits,
#                               both_hit]))
#     head_tail_mrr = torch.stack(head_tail_mrr, 0)
#     torch.save(head_tail_mrr, out_dir + "rank_rel_eval.pt")


# def per_ent_eval(mapped_triples, file_triples: [], pred_scores, out_dir):
#     original_triples_df = pd.DataFrame(data=mapped_triples.numpy(), columns=['h', 'r', 't'])
#     original_groups_h = original_triples_df.groupby('h', group_keys=True, as_index=False)
#     original_groups_t = original_triples_df.groupby('t', group_keys=True, as_index=False)
#     ordered_keys_h = original_groups_h.groups.keys()
#     ordered_keys_t = original_groups_t.groups.keys()
#     triples_df = pd.DataFrame(data=file_triples, columns=['h', 'r', 't'])
#     h_groups = triples_df.groupby('h', group_keys=True, as_index=False)
#     h_ent2idx = {key: idx for idx, key in enumerate(ordered_keys_h)}
#     t_groups = triples_df.groupby('t', group_keys=True, as_index=False)
#     t_ent2idx = {key: idx for idx, key in enumerate(ordered_keys_t)}
#     h_ent_eval = []
#     t_ent_eval = []
#     for h in ordered_keys_h:
#         if h not in h_groups.groups.keys():
#             h_ent_eval.append(0)
#         else:
#             rg_tris = h_groups.get_group(h)
#             rg_index = torch.as_tensor(rg_tris.index)
#             h_preds, t_preds = pred_scores[rg_index].chunk(2, 1)
#             t_preds = t_preds.squeeze(1)
#             tails = torch.as_tensor(rg_tris['t'].values)
#             tail_hits = calc_hit_at_k(t_preds, tails)
#             h_ent_eval.append(torch.as_tensor(tail_hits))
#     for t in ordered_keys_t:
#         if t not in t_groups.groups.keys():
#             t_ent_eval.append(0)
#         else:
#             rg_tris = t_groups.get_group(t)
#             rg_index = torch.as_tensor(rg_tris.index)
#             h_preds, t_preds = pred_scores[rg_index].chunk(2, 1)
#             h_preds = h_preds.squeeze(1)
#             heads = torch.as_tensor(rg_tris['h'].values)
#             head_hits = calc_hit_at_k(h_preds, heads)
#             t_ent_eval.append(torch.as_tensor(head_hits))
#     torch.save(torch.stack(h_ent_eval, 0), out_dir + "rank_h_ent_eval.pt")
#     torch.save(torch.stack(t_ent_eval, 0), out_dir + "rank_t_ent_eval.pt")
#     save2json(h_ent2idx, out_dir + "rank_h_ent2idx.json")
#     save2json(t_ent2idx, out_dir + "rank_t_ent2idx.json")


def get_index_of_a_in_b(a, b):
    b_df = pd.DataFrame(data=b.numpy(), columns=['h', 'r', 't'])
    a_in_b_index = []
    for row in a.numpy():
        h = row[0]
        r = row[1]
        t = row[2]
        re = b_df.query("h==@h & r==@r & t==@t")
        a_in_b_index.append(re.index.values[0])
    return a_in_b_index


def mapped_triples_2_anyburl_hrt_dev(dataset: pykeen.datasets.Dataset, anyburl_dir):
    init_dir(anyburl_dir)
    init_dir(anyburl_dir + 'predictions/')
    init_dir(anyburl_dir + 'rules/')
    clean_anyburl_tmp_files(anyburl_dir)
    df_train = pd.DataFrame(dataset.training.mapped_triples.numpy()).astype('int64')
    df_train.to_csv(osp.join(anyburl_dir, f'train.txt'), header=False, index=False, sep='\t')
    df_dev = pd.DataFrame(dataset.validation.mapped_triples.numpy()).astype('int64')
    df_dev.to_csv(osp.join(anyburl_dir, f'valid.txt'), header=False, index=False, sep='\t')
    # use dev as blender training set
    df_test = pd.DataFrame(dataset.validation.mapped_triples.numpy()).astype('int64')
    df_test.to_csv(osp.join(anyburl_dir, f'test.txt'), header=False, index=False, sep='\t')
    wait_until_anyburl_data_ready(anyburl_dir)


def mapped_test_2_anyburl_hrt_test(dataset: pykeen.datasets.Dataset, anyburl_dir):
    clean_anyburl_tmp_files(anyburl_dir)
    df_test = pd.DataFrame(dataset.testing.mapped_triples.numpy()).astype('int64')
    df_test.to_csv(osp.join(anyburl_dir, f'test.txt'), header=False, index=False, sep='\t')
    wait_until_anyburl_data_ready(anyburl_dir)


def prepare_anyburl_configs(anyburl_dir, snapshot=100, top_k=10):
    config_apply = f"PATH_TRAINING  = {anyburl_dir}train.txt\n" \
                   f"PATH_TEST      = {anyburl_dir}test.txt\n" \
                   f"PATH_VALID     = {anyburl_dir}valid.txt\n" \
                   f"PATH_RULES     = {anyburl_dir}rules/alpha-{snapshot}\n" \
                   f"PATH_OUTPUT    = {anyburl_dir}predictions/alpha-{snapshot}\n" \
                   "UNSEEN_NEGATIVE_EXAMPLES = 5\n" \
                   f"TOP_K_OUTPUT = {top_k}\n" \
                   "WORKER_THREADS = 7"
    config_eval = f"PATH_TRAINING  = {anyburl_dir}train.txt\n" \
                  f"PATH_TEST      = {anyburl_dir}test.txt\n" \
                  f"PATH_VALID     = {anyburl_dir}valid.txt\n" \
                  f"PATH_PREDICTIONS   = {anyburl_dir}predictions/alpha-{snapshot}"
    config_learn = f"PATH_TRAINING  = {anyburl_dir}train.txt\n" \
                   f"PATH_OUTPUT    = {anyburl_dir}rules/alpha\n" \
                   f"SNAPSHOTS_AT = {snapshot}\n" \
                   f"WORKER_THREADS = 4\n"
    save_to_file(config_apply, anyburl_dir + "config-apply.properties")
    save_to_file(config_eval, anyburl_dir + "config-eval.properties")
    save_to_file(config_learn, anyburl_dir + "config-learn.properties")
    wait_until_file_is_saved(anyburl_dir + "config-apply.properties")
    wait_until_file_is_saved(anyburl_dir + "config-eval.properties")
    wait_until_file_is_saved(anyburl_dir + "config-learn.properties")


def clean_anyburl_tmp_files(anyburl_dir):
    init_dir(f"{anyburl_dir}last_round/")
    os.system(f"[ -f {anyburl_dir}predictions/alpha* ] && mv {anyburl_dir}predictions/* {anyburl_dir}last_round/")
    os.system(f"[ -f {anyburl_dir}anyburl_eval.log ] && rm {anyburl_dir}anyburl_eval.log")
    os.system(f"[ -f {anyburl_dir}test.txt ] && rm {anyburl_dir}test.txt")


def wait_until_anyburl_data_ready(anyburl_dir):
    for f in ['train', 'valid', 'test']:
        wait_until_file_is_saved(anyburl_dir + "{}.txt".format(f))


def learn_anyburl(work_dir, snapshot=100):
    if work_dir[-1] == '/':
        work_dir = work_dir[:-1]
    os.system('./run_anyburl.sh ' + work_dir)
    wait_until_file_is_saved(work_dir + f"/rules/alpha-{snapshot}", 60)


def predict_with_anyburl(work_dir, snapshot=100):
    print("Predicting via AnyBURL...")
    os.system(f"java -Xmx10G -cp {work_dir}AnyBURL-JUNO.jar de.unima.ki.anyburl.Apply {work_dir}config-apply.properties")
    wait_until_file_is_saved(work_dir + f"predictions/alpha-{snapshot}_plog", 60)
    print("Evaluating AnyBURL...")
    os.system(f"java -Xmx10G -cp {work_dir}AnyBURL-JUNO.jar de.unima.ki.anyburl.Eval {work_dir}config-eval.properties > {work_dir}anyburl_eval.log")


def clean_anyburl(work_dir):
    if work_dir[-1] == '/':
        work_dir = work_dir[:-1]
    os.system('clean_anyburl.sh ' + work_dir)


class LpAnyBURL:
    def __init__(self, params):
        self.dataset = get_dataset(dataset=params.dataset)
        self.work_dir = params.work_dir
        self.snapshot = params.snapshot
        self.top_k = params.top_k

    def dev_pred(self):
        mapped_triples_2_anyburl_hrt_dev(self.dataset, self.work_dir)
        prepare_anyburl_configs(self.work_dir, snapshot=self.snapshot, top_k=self.top_k)
        learn_anyburl(self.work_dir, snapshot=self.snapshot)
        predict_with_anyburl(self.work_dir, snapshot=self.snapshot)
        pykeen_formated = anyburl_to_pykeen_format(self.dataset.validation.mapped_triples,
                                                   self.dataset.num_entities,
                                                   anyburl_dir=self.work_dir,
                                                   snapshot=self.snapshot,
                                                   top_k=self.top_k)
        all_pos_triples = get_all_pos_triples(self.dataset)
        to_fusion_eval_format(self.dataset, pykeen_formated, all_pos_triples, self.work_dir, self.top_k)
        # per_rel_eval(self.dataset.validation.mapped_triples, pykeen_formated, self.work_dir)
        # per_mapping_eval(dataset=self.dataset, pred_scores=pykeen_formated, out_dir=self.work_dir)
        # per_ent_eval(self.dataset.validation.mapped_triples, file_triples, pred_scores, self.work_dir)

    def test_pred(self):
        clean_anyburl_tmp_files(self.work_dir)
        mapped_test_2_anyburl_hrt_test(self.dataset, self.work_dir)
        predict_with_anyburl(self.work_dir, snapshot=self.snapshot)
        pykeen_formated = anyburl_to_pykeen_format(self.dataset.testing.mapped_triples,
                                                   num_candidates=self.dataset.num_entities,
                                                   anyburl_dir=self.work_dir,
                                                   snapshot=self.snapshot,
                                                   top_k=self.top_k)
        torch.save(pykeen_formated, self.work_dir + "preds.pt")

    def train_and_pred(self):
        self.dev_pred()
        self.test_pred()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument('--snapshot', type=int, default=100)
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/anyburl/")
    args = parser.parse_args()
    lp = LpAnyBURL(args)
    lp.train_and_pred()

