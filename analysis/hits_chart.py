import argparse
from collections import Counter
import numpy as np
import pandas as pd
import torch
from pykeen.constants import TARGET_TO_INDEX
from pykeen.datasets import get_dataset
from pykeen.typing import LABEL_HEAD, LABEL_TAIL, MappedTriples, Target
from torch import FloatTensor
from analysis.group_eval_utils import mask_positives
from context_load_and_run import load_score_context
from lp_kge.lp_pykeen import get_all_pos_triples
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class HitsChart():
    def __init__(self, params):
        self.params = params
        self.dataset = get_dataset(
            dataset=params.dataset
        )

    def _get_triples_per_hits(self,
                              batch: MappedTriples,
                              scores: FloatTensor,
                              target: Target,):
        # [[hit@1 triples], [hit@3 triples], [hit@10 triples]]
        column = TARGET_TO_INDEX[target]
        # # Select scores of true
        true_scores = scores[torch.arange(0, batch.shape[0]), batch[:, column]]
        ranks = torch.zeros(scores.shape[0])
        ranks[:] += torch.sum((scores >= true_scores.unsqueeze(1)).float(), dim=1).cpu()
        hits_at = list(map(lambda x: (ranks <= x).nonzero().squeeze(), [1, 3, 10]))
        non_hits_at = list(map(lambda x: (ranks > x).nonzero().squeeze(), [1, 3, 10]))
        return hits_at, non_hits_at

    def _read_hits_for_models(self):
        # return list format as [[hit1_tris, hit3_tris, hit10_tris], [hit1, hit3, hit10]...]
        context = load_score_context(self.params.models,
                                     in_dir=self.params.work_dir,
                                     calibration=self.params.cali == 'True'
                                     )
        all_pos_triples = get_all_pos_triples(self.dataset)
        hits = []
        non_hits = []
        for m in self.params.models:
            m_pred = context[m]['preds']
            h_preds, t_preds = torch.chunk(torch.as_tensor(m_pred), 2, 1)
            # restore format that required by pykeen evaluator
            ht_scores = [h_preds, t_preds]
            ht_hits = []
            non_ht_hits = []
            for ind, target in enumerate([LABEL_HEAD, LABEL_TAIL]):
                # mask pos scores
                masked_ht_scores = mask_positives(self.dataset.testing.mapped_triples, ht_scores[ind], target, all_pos_triples, -999)
                hits_at, non_hits_at = self._get_triples_per_hits(
                    batch=self.dataset.testing.mapped_triples,
                    scores=masked_ht_scores,
                    target=target,
                )
                ht_hits.append(hits_at)
                non_ht_hits.append(non_hits_at)
            hits.append(ht_hits)
            non_hits.append(non_ht_hits)
        return hits, non_hits

    def _to_sort_dict(self, model_hits):
        # {'h': {by1model:{hit@1: [id1, id2..], hit@3: [], hit@10: []}, by2models: ...}
        hit_at = [1, 3, 10]
        ht_hits_dict = dict()
        for target, h_or_t in enumerate(['h', 't']):
            # head or tail prediction
            all_target_hits = [m_hits[target] for m_hits in model_hits]
            # all hits in models
            all_target_hits = [torch.cat(
                [m_target_hits[idx] for m_target_hits in all_target_hits if len(m_target_hits[idx].shape) != 0])
                               for idx, hitK in enumerate(hit_at)]
            hit_at_dict = dict()
            for idx, k in enumerate(hit_at):
                count_dict = dict()
                df_hit_k_counts = pd.DataFrame(Counter(all_target_hits[idx].tolist()).most_common(),
                                               columns=['id', 'count'])
                for group in df_hit_k_counts.groupby('count', group_keys=True, as_index=False):
                    hit_count = group[0]
                    hit_ids = group[1]['id'].to_list()
                    count_dict.update({hit_count: hit_ids})
                hit_at_dict.update({k: count_dict})
            ht_hits_dict.update({h_or_t: hit_at_dict})
        return ht_hits_dict

    def to_charts(self, hits_dict, non_hits_dict):
        fig, axs = plt.subplots(1, 2, sharey=True)
        x_num = len(self.params.models)
        total_num = self.dataset.testing.num_triples
        x = np.arange(0, x_num + 1)
        colors = ['red', 'green', 'orange']
        for idx, target in enumerate(['h', 't']):
            target_dict = hits_dict[target]
            target_dict_non = non_hits_dict[target]
            for hit_idx, hit in enumerate([1, 3, 10]):
                # preparing y data
                tmp_y_data = target_dict[hit]
                y = np.zeros([x_num + 1, ])
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
        plt.savefig(self.params.work_dir + 'model_hits.png', dpi=600)
        fig.show()

    def analyze(self):
        hits, non_hits = self._read_hits_for_models()
        hit_dict = self._to_sort_dict(hits)
        non_hit_dict = self._to_sort_dict(non_hits)
        self.to_charts(hit_dict, non_hit_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="CP_CPComplEx_TuckER_RotatE_anyburl")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    parser.add_argument('--cali', type=str, default="True")
    args = parser.parse_args()
    args.models = args.models.split('_')
    model_hits_at = HitsChart(params=args)
    model_hits_at.analyze()
