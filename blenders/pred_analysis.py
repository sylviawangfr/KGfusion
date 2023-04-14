import argparse
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import torch
from pykeen.constants import TARGET_TO_INDEX
from pykeen.datasets import get_dataset
from pykeen.evaluation import RankBasedEvaluator
from pykeen.evaluation.evaluator import filter_scores_, create_sparse_positive_filter_
from pykeen.typing import LABEL_HEAD, LABEL_TAIL, MappedTriples, Target
from pykeen.utils import resolve_device
from torch import FloatTensor
from blenders.blender_utils import eval_with_blender_scores, Blender
from common_utils import format_result, save_to_file
from context_load_and_run import load_score_context
from features.feature_scores_only_dataset import ScoresOnlyDataset
from lp_kge.lp_pykeen import get_all_pos_triples, predict_head_tail_scores
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def read_hits_for_models(params):
    context = load_score_context(params.models,
                                      in_dir=params.work_dir,
                                      calibration=params.cali == 'True'
                                      )
    ds = get_dataset(
        dataset=params.dataset
    )
    all_pos_triples = get_all_pos_triples(ds)
    hits = []
    non_hits = []
    for m in params.models:
        m_pred = context[m]['preds']
        h_preds, t_preds = torch.chunk(torch.as_tensor(m_pred), 2, 1)
        # restore format that required by pykeen evaluator
        ht_scores = [h_preds, t_preds]
        ht_hits = []
        non_ht_hits = []
        for ind, target in enumerate([LABEL_HEAD, LABEL_TAIL]):
            hits_at, non_hits_at = get_triples_per_hits(
                batch=ds.testing.mapped_triples,
                scores=ht_scores[ind],
                target=target,
                all_pos_triples=all_pos_triples,
            )
            ht_hits.append(hits_at)
            non_ht_hits.append(non_hits_at)
        hits.append(ht_hits)
        non_hits.append(non_ht_hits)
    return hits, non_hits, ds.testing.mapped_triples.shape[0]


def to_sort_dict(model_hits):
    hit_at = [1, 3, 10]
    ht_hits_dict = dict()
    for target, h_or_t in enumerate(['h', 't']):
        # head or tail prediction
        all_target_hits = [m_hits[target] for m_hits in model_hits]
        # all hits in models
        all_target_hits = [torch.cat([m_target_hits[idx] for m_target_hits in all_target_hits if len(m_target_hits[idx].shape)!=0])
                           for idx, hitK in enumerate(hit_at)]
        hit_at_dict = dict()
        for idx, k in enumerate(hit_at):
            count_dict = dict()
            df_hit_k_counts = pd.DataFrame(Counter(all_target_hits[idx].tolist()).most_common(), columns=['id', 'count'])
            for group in df_hit_k_counts.groupby('count', group_keys=True, as_index=False):
                hit_count = group[0]
                hit_ids = group[1]['id'].to_list()
                count_dict.update({hit_count: hit_ids})
            hit_at_dict.update({k: count_dict})
        ht_hits_dict.update({h_or_t: hit_at_dict})
    return ht_hits_dict


def analyze_hits(params):
    hits, non_hits, total_num = read_hits_for_models(params)
    hit_dict = to_sort_dict(hits)
    non_hit_dict = to_sort_dict(non_hits)
    draw_hits_charts(hit_dict, non_hit_dict, len(params.models), total_num, params.work_dir + "hits.png")


def draw_hits_charts(hits_dict, non_hits_dict, x_num, total_num, out_file):
    fig, axs = plt.subplots(1, 2, sharey=True)
    x = np.arange(0, x_num + 1)
    colors = ['red', 'green', 'orange']
    for idx, target in enumerate(['h', 't']):
        target_dict = hits_dict[target]
        target_dict_non = non_hits_dict[target]
        for hit_idx, hit in enumerate([1, 3, 10]):
            # preparing y data
            tmp_y_data = target_dict[hit]
            y = np.zeros([x_num + 1,])
            y[0] = len(target_dict_non[hit][x_num]) / total_num if x_num in target_dict_non[hit] else 0
            y[[i for i in list(tmp_y_data.keys())]] = [len(tmp_y_data[m]) / total_num for m in tmp_y_data.keys()]
            axs[idx].plot(x, y, color=colors[hit_idx], label=f'hit@{hit}')
        axs[idx].set_ylim((0, 1))
        axs[idx].set_title(f'{target} pred')

    plt.setp(axs, xticks=x, xticklabels=x)
    fig.legend(handles=[mpatches.Patch(color='red', label='hit@1'),
                        mpatches.Patch(color='green', label='hit@3'),
                        mpatches.Patch(color='orange', label='hit@10')],
               loc='upper right', ncol=3, bbox_to_anchor=(0.96, 0.92))
    axs[0].set_ylabel('num_hits / total_num')
    fig.tight_layout()
    plt.subplots_adjust(top=0.8, left=0.1)
    if len(out_file):
        plt.savefig(out_file, dpi=600)
    fig.show()


def get_triples_per_hits(batch: MappedTriples,
                         scores: FloatTensor,
                         target: Target,
                         all_pos_triples: Optional[MappedTriples],):
    column = TARGET_TO_INDEX[target]
    positive_filter, relation_filter = create_sparse_positive_filter_(
        hrt_batch=batch,
        all_pos_triples=all_pos_triples,
        relation_filter=None,
        filter_col=column,
    )
    # Select scores of true
    true_scores = scores[torch.arange(0, batch.shape[0]), batch[:, column]]
    # overwrite filtered scores
    scores = filter_scores_(scores=scores, filter_batch=positive_filter)
    # The scores for the true triples have to be rewritten to the scores tensor
    scores[torch.arange(0, batch.shape[0]), batch[:, column]] = true_scores
    # the rank-based evaluators needs the true scores with trailing 1-dim
    ranks = torch.zeros(scores.shape[0])
    ranks[:] += torch.sum((scores >= true_scores.unsqueeze(1)).float(), dim=1).cpu()
    hits_at = list(map(lambda x: (ranks <= x).nonzero().squeeze(), [1, 3, 10]))
    non_hits_at = list(map(lambda x: (ranks > x).nonzero().squeeze(), [1, 3, 10]))
    return hits_at, non_hits_at


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="CP_CPComplEx_TuckER_RotatE_anyburl")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    parser.add_argument('--cali', type=str, default="True")
    args = parser.parse_args()
    args.models = args.models.split('_')
    analyze_hits(args)
