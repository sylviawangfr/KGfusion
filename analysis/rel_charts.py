import argparse
from collections import Counter
from typing import Optional
import numpy as np
import pandas as pd
import torch
from pykeen.constants import TARGET_TO_INDEX
from pykeen.datasets import get_dataset
from pykeen.evaluation.evaluator import filter_scores_, create_sparse_positive_filter_
from pykeen.typing import LABEL_HEAD, LABEL_TAIL, MappedTriples, Target
from tabulate import tabulate
from torch import FloatTensor

from analysis.group_eval_utils import group_rank_eval, AnalysisChart
from lp_kge.lp_pykeen import find_relation_mappings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class RelChart(AnalysisChart):
    def __init__(self, params):
        super().__init__(params)

    def make_partitions(self, mapped_triples):
        triples_df = pd.DataFrame(data=mapped_triples.numpy(), columns=['h', 'r', 't'])
        original_groups = triples_df.groupby('r', group_keys=True, as_index=False)
        group_keys = original_groups.groups.keys()
        key2tri_ids = dict()
        for key in group_keys:
            g = original_groups.get_group(key)
            g_index = torch.from_numpy(g.index.values)
            key2tri_ids.update({key: g_index})
        return key2tri_ids

    def _to_chart(self, key2eval):
        pass

    def analyze(self):
        key2tri_ids = self.make_partitions(self.dataset.testing.mapped_triples)
        key2eval = group_rank_eval(self.dataset,
                                   self.dataset.testing.mapped_triples,
                                   key2tri_ids,
                                   self.context['preds'])


class RelMappingChart(AnalysisChart):
    def __init__(self, params):
        super().__init__(params)
        self.mappings = ['1-1', '1-n', 'n-1', 'n-m']

    def make_partitions(self, mapped_triples):
        rel_mappings = find_relation_mappings(self.dataset)
        triples_df = pd.DataFrame(data=mapped_triples.numpy(), columns=['h', 'r', 't'], )
        key2tri_ids = dict()
        for key in self.mappings:
            tmp_rels = rel_mappings[key]
            tri_group = triples_df.query('r in @tmp_rels')
            if len(tri_group.index) > 0:
                g_index = torch.from_numpy(tri_group.index.values)
                key2tri_ids.update({key: g_index})
        return key2tri_ids

    def get_partition_eval_per_model(self, key2tri_ids):
        m_dict = dict()
        for m in self.params.models:
            key2eval = group_rank_eval(self.dataset,
                                       self.dataset.testing.mapped_triples,
                                       key2tri_ids,
                                       self.context[m]['preds'])
            m_dict.update({m: key2eval})
        return m_dict

    def _to_bar_chart(self, m2eval):
        m2data = {}
        max_y = 0
        min_y = 1
        for m in self.params.models:
            m_data = []
            m_eval = m2eval[m]
            for t in self.mappings:
                if t in m_eval:
                    mrr = m_eval[t][-1, -1].item()
                    m_data.append(mrr)
                    max_y = mrr if mrr > max_y else max_y
                    min_y = mrr if mrr < min_y else min_y
                else:
                    m_data.append(0)
            m2data.update({m: m_data})
        x = np.arange(len(self.mappings))  # the label locations
        width = 0.05  # the width of the bars
        multiplier = 0
        fig, ax = plt.subplots(layout='constrained')
        for attribute, measurement in m2data.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute)
            # ax.bar_label(rects, padding=3)
            multiplier += 1
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('MRR')
        ax.set_title('MRR on relation mappings')
        ax.set_xticks(x + width, self.mappings)
        ax.legend(loc='upper left')
        min_y = 0.9 * min_y if 0.9 * min_y < 0.5 else 0.5
        max_y = 1.1 * max_y if 1.1 * max_y < 1 else 1
        ax.set_ylim(min_y, max_y)
        plt.show()

    def _to_pie_chart(self, key2tri_ids):
        labels = key2tri_ids.keys()
        values = [len(key2tri_ids[t]) for t in key2tri_ids]
        fig, ax = plt.subplots()
        ax.pie(values, labels=labels)
        plt.show()

    def _to_table(self, m2eval):
        header2 = ['', '1-1\nh', '1-1\nt', '1-n\nh', '1-n\nt', 'n-1\nh', 'n-1\nt', 'n-m\nh', 'n-m\nt']
        data = []
        for m in self.params.models:
            m_data = [m]
            m_eval = m2eval[m]
            for t in self.mappings:
                if t in m_eval:
                    m_data.extend([m_eval[t][0, -1].item(), m_eval[t][1, -1].item()])
                else:
                    m_data.extend([0, 0])
            data.append(m_data)
        print(tabulate(data, header2, tablefmt="simple_grid", numalign="center"))

    def analyze(self):
        key2tri_ids = self.make_partitions(self.dataset.testing.mapped_triples)
        m2eval = self.get_partition_eval_per_model(key2tri_ids)
        self._to_pie_chart(key2tri_ids)
        self._to_bar_chart(m2eval)
        self._to_table(m2eval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="CP_CPComplEx_TuckER_RotatE_anyburl")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    parser.add_argument('--cali', type=str, default="True")
    # parser.add_argument('--partition', type=str, default="rel_mapping")
    args = parser.parse_args()
    args.models = args.models.split('_')
    tmp = RelMappingChart(args)
    tmp.analyze()
