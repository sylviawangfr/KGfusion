import os
import os.path as osp
from itertools import zip_longest
import pandas as pd
import pykeen.datasets
import torch
from pykeen.datasets import Nations

from common_utils import save_to_file, wait_until_file_is_saved, init_dir


def to_pykeen_eval_format(pred_index, pred_scores):
    pass


def read_hrt_pred_anyburl(pred_anyburl_file, top_k=10):
    with open(pred_anyburl_file) as f:
        lines = f.readlines()
        chunks = zip_longest(*[iter(lines)] * 3, fillvalue='')
        h_preds = []
        t_preds = []
        for chunk in chunks:
            hs = chunk[1][7:].strip().split('\t')
            h_preds.extend(torch.as_tensor([float(hs[i]) if i < len(hs) else -1 for i in range(top_k * 2)]))
            ts = chunk[2][7:].strip().split('\t')
            t_preds.extend(torch.as_tensor([float(ts[i]) if i < len(ts) else -1 for i in range(top_k * 2)]))
        h_preds = torch.cat(h_preds, 0)
        preserve_shape = h_preds.shape
        h_preds = h_preds.reshape([preserve_shape[0] * top_k, 2])
        h_index = h_preds[:, 0].reshape([preserve_shape[0], top_k])
        h_scores = h_preds[:, 1].reshape([preserve_shape[0], top_k])
        t_preds = t_preds.reshape([preserve_shape[0] * top_k, 2])
        t_index = t_preds[:, 0].reshape([preserve_shape[0], top_k])
        t_scores = t_preds[:, 1].reshape([preserve_shape[0], top_k])
        pred_index = torch.cat([h_index, t_index], 1)
        pred_scores = torch.cat([h_scores, t_scores], 1)
    return pred_index, pred_scores


def mapped_triples_2_anyburl_hrt(dataset: pykeen.datasets.Dataset, anyburl_dir):
    init_dir(anyburl_dir)
    df_train = pd.DataFrame(dataset.training.mapped_triples.numpy()).astype('int64')
    df_train.to_csv(osp.join(anyburl_dir, f'train.txt'), header=False, index=False, sep='\t')
    df_dev = pd.DataFrame(dataset.validation.mapped_triples.numpy()).astype('int64')
    df_dev.to_csv(osp.join(anyburl_dir, f'valid.txt'), header=False, index=False, sep='\t')
    df_test = pd.DataFrame(dataset.testing.mapped_triples.numpy()).astype('int64')
    df_test.to_csv(osp.join(anyburl_dir, f'test.txt'), header=False, index=False, sep='\t')
    wait_until_anyburl_data_ready(anyburl_dir)


def prepare_anyburl_configs(anyburl_dir):
    config_apply = f"PATH_TRAINING  = {anyburl_dir}train.txt\n" \
                   f"PATH_TEST      = {anyburl_dir}test.txt\n" \
                   f"PATH_VALID     = {anyburl_dir}valid.txt\n" \
                   f"PATH_RULES     = {anyburl_dir}rules/alpha-100\n" \
                   f"PATH_OUTPUT    = {anyburl_dir}predictions/alpha-100\n" \
                   "UNSEEN_NEGATIVE_EXAMPLES = 5\n" \
                   "TOP_K_OUTPUT = 10\n" \
                   "WORKER_THREADS = 7"
    config_eval = f"PATH_TRAINING  = {anyburl_dir}train.txt\n" \
                  f"PATH_TEST      = {anyburl_dir}test.txt\n" \
                  f"PATH_VALID     = {anyburl_dir}valid.txt\n" \
                  f"PATH_PREDICTIONS   = {anyburl_dir}predictions/alpha-100"
    config_learn = f"PATH_TRAINING  = {anyburl_dir}train.txt\n" \
                   f"PATH_OUTPUT    = {anyburl_dir}rules/alpha\n" \
                   f"SNAPSHOTS_AT = 100\n" \
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
    os.system(f"[ -f {anyburl_dir}config-apply.properties ] && mv {anyburl_dir}config-apply.properties {anyburl_dir}last_round/")
    os.system(f"[ -f {anyburl_dir}anyburl_eval.log ] && rm {anyburl_dir}anyburl_eval.log")


def wait_until_anyburl_data_ready(anyburl_dir):
    for f in ['train', 'valid', 'test']:
        wait_until_file_is_saved(anyburl_dir + "{}.txt".format(f))


def learn_anyburl(work_dir):
    if work_dir[-1] == '/':
        work_dir = work_dir[:-1]
    os.system('../anyBURL/run_anyburl.sh ' + work_dir)
    wait_until_file_is_saved(work_dir + "/rules/alpha-100", 60)


def predict_with_anyburl(work_dir):
    print("Predicting via AnyBURL...")
    os.system(f"java -Xmx10G -cp {work_dir}AnyBURL-JUNO.jar de.unima.ki.anyburl.Apply {work_dir}config-apply.properties")
    wait_until_file_is_saved(work_dir + "predictions/alpha-100_plog", 60)
    print("Evaluating AnyBURL...")
    os.system(f"java -Xmx10G -cp {work_dir}AnyBURL-JUNO.jar de.unima.ki.anyburl.Eval {work_dir}config-eval.properties > {work_dir}anyburl_eval.log")


def read_eval_result(work_dir):
    with open(work_dir + "anyburl_eval.log") as f:
        last_line = f.readlines()[-1].strip()
        scores = last_line.split()
        formated = f"hit@1, hit@3, hit@10, MRR\n" + "'".join(scores)
        return formated


def clean_anyburl(work_dir):
    if work_dir[-1] == '/':
        work_dir = work_dir[:-1]
    os.system('../scripts/clean_anyburl.sh ' + work_dir)


def anyburl_pipeline(dataset, workdir):
    # mapped_triples_2_anyburl_hrt(dataset, workdir)
    # prepare_anyburl_configs(workdir)
    # learn_anyburl(workdir)
    # predict_with_anyburl(workdir)
    df = read_hrt_pred_anyburl(workdir + "predictions/alpha-100")
    to_pykeen_pred_format(df)



if __name__ == "__main__":
    # mapped_triples_2_anyburl_hrt(Nations(), '../outputs/nations/anyburl/')
    anyburl_pipeline(Nations(), '../outputs/nations/anyburl/')
