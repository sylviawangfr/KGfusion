import argparse
import os
import os.path as osp
from itertools import zip_longest
import pykeen.datasets
import torch
import pandas as pd
from pykeen.datasets import FB15k237, UMLS, get_dataset

from common_utils import save_to_file, wait_until_file_is_saved, init_dir
from lp_kge.lp_pykeen import get_neg_scores_top_k, get_all_pos_triples, find_relation_mappings


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
    return file_triples, pred_index, pred_scores


def to_fusion_eval_format(dataset: pykeen.datasets.Dataset, file_triples: [], pred_index, pred_scores, all_pos_triples, out_dir, top_k=10):
    # anyburl doesn't predict in order of input triples, we need sort result to compat with pykeen and reuse pykeen's api to filter all pos triples
    mapped_triples = dataset.validation.mapped_triples
    num_candidates = dataset.num_entities
    pred_index[pred_index == -1] = num_candidates  # add one extra column to handle mask value/index
    pykeen_scatter_scores = torch.zeros([pred_scores.shape[0], 2, num_candidates + 1]).scatter_(2, pred_index, pred_scores) # sigmoid scores [0-1]
    # get index of triples
    file_index = get_certain_index_in_tensor(mapped_triples, file_triples)
    eval_preds = torch.zeros([mapped_triples.shape[0], 2, num_candidates])
    eval_preds[file_index] = pykeen_scatter_scores[:, :, :-1]
    m_dev_preds = torch.chunk(eval_preds, 2, 1)
    m_dev_preds = [i.squeeze(1) for i in m_dev_preds]
    pos_scores = m_dev_preds[0]
    pos_scores = pos_scores[torch.arange(0, dataset.validation.mapped_triples.shape[0]),
                            dataset.validation.mapped_triples[:, 0]]
    neg_scores, neg_index_topk = get_neg_scores_top_k(dataset.validation.mapped_triples, m_dev_preds, all_pos_triples, top_k) # [[h1 * candidate, h2 * candicate...][t1,t2...]]
    torch.save(pos_scores, out_dir + "eval_pos_scores.pt")
    torch.save(neg_scores, out_dir + "eval_neg_scores.pt")
    torch.save(neg_index_topk, out_dir + "eval_neg_index.pt")


def to_fusion_test_format(dataset, file_triples: [], pred_index, pred_scores, out_dir):
    # anyburl doesn't predict in order of input triples, we need sort result to compat with pykeen
    mapped_triples = dataset.testing.mapped_triples
    num_candidates = dataset.num_entities
    # get index of triples
    file_index = get_certain_index_in_tensor(mapped_triples, file_triples)
    pred_index[pred_index == -1] = num_candidates  # add one extra column to handle mask value/index
    pykeen_scatter_scores = torch.zeros([pred_scores.shape[0], 2, num_candidates + 1]).scatter_(2, pred_index, pred_scores) # sigmoid scores [0-1]
    scores = torch.zeros([mapped_triples.shape[0], 2, num_candidates])
    scores[file_index] = pykeen_scatter_scores[:, :, :-1]
    scores = torch.chunk(scores, 2, 1)
    scores = torch.cat([i.squeeze(1) for i in scores], 1)
    torch.save(scores, out_dir + "preds.pt")


def per_rel_eval(mapped_triples, file_triples: [], pred_index, out_dir):
    tri_df = pd.DataFrame(data=file_triples, columns=['h', 'r', 't'])
    pred_rel_groups = tri_df.groupby('r', group_keys=True, as_index=False)
    hit_positions = [1, 3, 10]
    triples_df = pd.DataFrame(data=mapped_triples.numpy(), columns=['h', 'r', 't'])
    original_groups = triples_df.groupby('r', group_keys=True, as_index=False)
    ordered_keys = original_groups.groups.keys()
    head_tail_mrr = []
    for rel in ordered_keys:
        if rel not in pred_rel_groups.groups.keys():
            head_tail_mrr.append([0, 0, 0])
        else:
            rg_tris = pred_rel_groups.get_group(rel)
            rg_index = torch.as_tensor(rg_tris.index)
            h_preds, t_preds = pred_index[rg_index].chunk(2, 1)
            h_preds = h_preds.squeeze(1)
            t_preds = t_preds.squeeze(1)
            heads = torch.as_tensor(rg_tris['h'].values).unsqueeze(1)
            tails = torch.as_tensor(rg_tris['t'].values).unsqueeze(1)
            head_hits = calc_hit_at_10(h_preds, heads)
            tail_hits = calc_hit_at_10(t_preds, tails)
            both_hit = calc_hit_at_10(torch.cat([h_preds, t_preds]), torch.cat((heads, tails)))
            head_tail_mrr.append([head_hits,
                                  tail_hits,
                                  both_hit])
    head_tail_mrr = torch.Tensor(head_tail_mrr)
    torch.save(head_tail_mrr, out_dir + "rel_eval.pt")


def per_mapping_eval(dataset, file_triples: [], pred_index, out_dir):
    mappings = ['one_to_one', 'one_to_many', 'many_to_one', 'many_to_many']
    rel_mappings = find_relation_mappings(dataset)
    tri_df = pd.DataFrame(data=file_triples, columns=['h', 'r', 't'])
    head_tail_mrr = []
    for rel_group in mappings:
        tmp_rels = rel_mappings[rel_group]
        rg_tris = tri_df.query('r in @tmp_rels')
        if len(rg_tris.index) > 0:
            rg_index = torch.as_tensor(rg_tris.index)
            h_preds, t_preds = pred_index[rg_index].chunk(2, 1)
            h_preds = h_preds.squeeze(1)
            t_preds = t_preds.squeeze(1)
            heads = torch.as_tensor(rg_tris['h'].values).unsqueeze(1)
            tails = torch.as_tensor(rg_tris['t'].values).unsqueeze(1)
            head_hits = calc_hit_at_10(h_preds, heads)
            tail_hits = calc_hit_at_10(t_preds, tails)
            both_hit = calc_hit_at_10(torch.cat([h_preds, t_preds]), torch.cat((heads, tails)))
            head_tail_mrr.append([head_hits,
                                  tail_hits,
                                  both_hit])
        else:
            head_tail_mrr.append([0.01, 0.01, 0.01])
    head_tail_mrr = torch.Tensor(head_tail_mrr)
    torch.save(head_tail_mrr, out_dir + "mapping_eval.pt")


def calc_hit_at_10(pred_idx, ground_truth_idx):
    """Calculates mean number of hits@k. Higher values are ranked first.
    Returns: list of float, of the same length as hit_positions, containing
        Hits@K score.
    """
    idx_at_10 = pred_idx[:, :10]
    hits_at_10 = (idx_at_10 == ground_truth_idx).sum(dim=1).float().mean()
    return hits_at_10.item()


def get_certain_index_in_tensor(src_tensor, target):
    df_src = pd.DataFrame(data=src_tensor.numpy())
    df_tgt = pd.DataFrame(data=target)
    diff1 = pd.concat([df_src, df_tgt]).drop_duplicates(keep=False)
    diff2 = pd.concat([df_src, diff1]).drop_duplicates(keep=False)
    tgt_idx = diff2.index
    return torch.as_tensor(tgt_idx)


def mapped_triples_2_anyburl_hrt_dev(dataset: pykeen.datasets.Dataset, anyburl_dir):
    init_dir(anyburl_dir)
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
    def __init__(self, dataset, work_dir):
        self.dataset = get_dataset(dataset=dataset)
        self.work_dir = work_dir

    def dev_pred(self, snapshot, top_k):
        mapped_triples_2_anyburl_hrt_dev(self.dataset, self.work_dir)
        prepare_anyburl_configs(self.work_dir, snapshot=snapshot, top_k=top_k)
        learn_anyburl(self.work_dir, snapshot=snapshot)
        predict_with_anyburl(self.work_dir, snapshot=snapshot)
        file_triples, pred_index, pred_scores = read_hrt_pred_anyburl(self.work_dir, snapshot=snapshot, top_k=top_k)
        all_pos_triples = get_all_pos_triples(self.dataset)
        to_fusion_eval_format(self.dataset, file_triples, pred_index, pred_scores, all_pos_triples, self.work_dir, top_k)
        per_rel_eval(self.dataset.validation.mapped_triples, file_triples, pred_index, self.work_dir)
        per_mapping_eval(dataset=self.dataset, file_triples=file_triples, pred_index=pred_index, out_dir=self.work_dir)

    def test_pred(self, snapshot, top_k):
        clean_anyburl_tmp_files(self.work_dir)
        mapped_test_2_anyburl_hrt_test(self.dataset, self.work_dir)
        predict_with_anyburl(self.work_dir, snapshot=snapshot)
        file_triples, pred_index, pred_scores = read_hrt_pred_anyburl(self.work_dir, snapshot=snapshot, top_k=top_k)
        to_fusion_test_format(self.dataset, file_triples, pred_index, pred_scores, self.work_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument('--snapshot', type=int, default=1000)
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/anyburl/")
    args = parser.parse_args()
    param1 = args.__dict__
    lp = LpAnyBURL(param1['dataset'], param1['work_dir'])
    lp.dev_pred(snapshot=param1['snapshot'], top_k=param1['top_k'])
    lp.test_pred(snapshot=param1['snapshot'], top_k=param1['top_k'])

