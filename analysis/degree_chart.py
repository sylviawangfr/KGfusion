import argparse
from collections import OrderedDict
from typing import Optional
import numpy as np
import pandas as pd
import torch
from pykeen.constants import TARGET_TO_INDEX
from pykeen.datasets import get_dataset
from pykeen.evaluation.evaluator import filter_scores_, create_sparse_positive_filter_
from pykeen.typing import LABEL_HEAD, LABEL_TAIL, MappedTriples, Target
from torch import FloatTensor
from itertools import zip_longest, chain
from analysis.group_eval_utils import group_rank_eval, AnalysisChart
from context_load_and_run import load_score_context
from lp_kge.lp_pykeen import get_all_pos_triples, find_relation_mappings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors


def find_degree_groups(dataset):
    # {in-degree: {0: [triple ids], 1: [], ...}
    # out-degree: {0: [triple ids], 1: [], ...}
    # degree: {0: [triple ids], 1: [], ...} }
    all_triples = torch.cat([dataset.training.mapped_triples,
                             dataset.validation.mapped_triples,
                             dataset.testing.mapped_triples], 0)
    df = pd.DataFrame(data=all_triples.numpy(), columns=['h', 'r', 't'])
    del all_triples
    all_ent_ids = pd.DataFrame(list(dataset.entity_to_id.values()), columns=['ent'])
    # calculate out degree
    ent_out = df.groupby('h', group_keys=True, as_index=False).size()  # number of out degree
    out_degree_ents = ent_out.groupby('size', as_index=True).agg(list)
    out2ents = OrderedDict(out_degree_ents.to_dict()['h'])
    # have to count degree 0
    all_head_ids = df['h'].drop_duplicates(keep='first')
    zero_degree_ids = pd.concat([all_ent_ids['ent'], all_head_ids], axis=0). \
        drop_duplicates(keep=False).values.tolist()
    out2ents.update({0: zero_degree_ids})
    out2ents.move_to_end(0, last=False)
    # calculate in degree
    ent_in = df.groupby('t', group_keys=True, as_index=False).size()  # number of out degree
    in_degree_ents = ent_in.groupby('size', as_index=True).agg(list)
    in2ents = OrderedDict(in_degree_ents.to_dict()['t'])
    # have to count degree 0
    all_tail_ids = df['t'].drop_duplicates(keep='first')
    zero_degree_ids = pd.concat([all_ent_ids['ent'], all_tail_ids], axis=0). \
        drop_duplicates(keep=False).values.tolist()
    in2ents.update({0: zero_degree_ids})
    in2ents.move_to_end(0, last=False)
    # count in and out
    ent_out = ent_out.rename(columns={'h': 'ent', 'size': 'out'})
    ent_in = ent_in.rename(columns={'t': 'ent', 'size': 'in'})
    ent_df = all_ent_ids.set_index('ent')
    ent_df = ent_df.join(ent_out.set_index('ent'), on='ent', how='left').fillna(0)
    ent_df = ent_df.join(ent_in.set_index('ent'), on='ent', how='left').fillna(0)
    ent_df['degree'] = ent_df['in'] + ent_df['out']
    ent_df = ent_df.drop(['in', 'out'], axis=1).astype(int)
    ent_df.reset_index(inplace=True)
    ent_df = ent_df.groupby('degree', as_index=True).agg(list)
    degree2ents = OrderedDict(ent_df.to_dict()['ent'])
    # count zero degree ents
    all_tri_ents = pd.concat([df['h'], df['t']], axis=0).drop_duplicates(keep='first')
    zero_degree_ids = pd.concat([all_ent_ids['ent'], all_tri_ents], axis=0). \
        drop_duplicates(keep=False).values.tolist()
    degree2ents.update({0: zero_degree_ids})
    degree2ents.move_to_end(0, last=False)
    # all directions
    result = dict()
    result.update({'In': in2ents, 'Out': out2ents, 'Entity': degree2ents})
    return result


class EntDegreeChart(AnalysisChart):
    def __init__(self, params):
        self.params = params
        self.dataset = get_dataset(
            dataset=params.dataset
        )
        self.context = load_score_context(self.params.models,
                                          in_dir=self.params.work_dir,
                                          calibration=self.params.cali == 'True'
                                          )
        self.all_pos_triples = get_all_pos_triples(self.dataset)

    def make_partitions(self, mapped_triples, target2degrees, ptt_num):
        # if the degree range is not suitable to draw as x-ticks, we can aggregate them to number of partitions.
        tri_df = pd.DataFrame(data=mapped_triples, columns=['h', 'r', 't'])
        query_strs = ["h in @ent_ids", "t in @ent_ids", "h in @ent_ids or t in @ent_ids"]
        target2tri_idx = dict()
        for i, (target, degree2ent_ids) in enumerate(target2degrees.items()):
            range_degree = range(0, max(list(degree2ent_ids.keys)) + 1)
            x_slots = np.array_split(range_degree, ptt_num)
            degrees2tri_idx = dict()
            for slot_degrees in x_slots:
                slot_entids = [degree2ent_ids[d] for d in slot_degrees]
                slot_entids = list(chain.from_iterable(slot_entids))
                slot_key = f"{slot_degrees[0]-slot_degrees[-1]}"
                tri_group = tri_df.query(query_strs[slot_entids])
                if len(tri_group.index) > 0:
                    g_index = torch.from_numpy(tri_group.index.values)
                    degrees2tri_idx.update({slot_key: g_index})
                else:
                    degrees2tri_idx.update({slot_key: []})
            target2tri_idx.update({target: degrees2tri_idx})
        return target2tri_idx

    def get_partition_eval(self, target2tri_idx):
        target2m2degree_eval = dict()
        for target, degree2tri_idx in target2tri_idx.items():
            m2eval = dict()
            for m in self.params.models:
                m_degree_group_eval = group_rank_eval(self.dataset,
                                                      self.dataset.testing.mapped_triples,
                                                      degree2tri_idx,
                                                      self.context[m]['preds'])
                m2eval.update({m: m_degree_group_eval})
            target2m2degree_eval.update({target: m2eval})
        return target2m2degree_eval

    def _to_degree_distribution_charts(self, target2degrees, y_lable='Entities'):
        for target, degree2ent_ids in target2degrees.items():
            degree = range(0, max(list(degree2ent_ids.keys())) + 1)
            numbers = np.zeros(len(degree))
            fill_idx = np.asarray(list(degree2ent_ids.keys()))
            fill_value = [len(degree2ent_ids[i]) for i in degree2ent_ids]
            numbers[fill_idx] = fill_value
            fig, ax = plt.subplots()
            ax.bar(degree, numbers)
            ax.set_ylabel(f'Number of {y_lable}')
            ax.set_xlabel(f'{target} Degree')
            ax.set_title(f'{target} Degree Distribution')
            plt.show()

    def _to_degree_eval_charts(self, target2m2degree_eval, target2degree2entids):
        to_pick_colors = list(mcolors.TABLEAU_COLORS.values())
        colors = {m: to_pick_colors[idx] for idx, m in enumerate(self.params.models)}
        for target, m2degree2eval in target2m2degree_eval.items():
            degrees = range(0, max(list(target2degree2entids[target].keys())) + 1)
            fig, ax = plt.subplots()
            # draw lines for all degree set
            for m, degree2eval in m2degree2eval.items():
                fill_idx = np.asarray(list(degree2eval.keys()))
                fill_value_h = [degree2eval[i][0][-1] for i in degree2eval]
                fill_value_t = [degree2eval[i][1][-1] for i in degree2eval]
                m_h_data = np.zeros(len(degrees))
                m_h_data[fill_idx] = fill_value_h
                m_t_data = np.zeros(len(degrees))
                m_t_data[fill_idx] = fill_value_t
                ax.plot(degrees, m_h_data, dashes=[4, 4], color=colors[m])
                ax.plot(degrees, m_t_data, color=colors[m])
            ax.set_ylabel('MRR')
            ax.set_xlabel(f'Number of {target}-degree')
            ax.set_title(f'MRR Eval on {target}-degree Sets')
            plt.show()

    def _to_table(self, key2eval):
        pass

    def analyze(self):
        target2degrees2entids = find_degree_groups(self.dataset)
        all_tris = torch.cat([self.dataset.testing.mapped_triples,
                              self.dataset.validation.mapped_triples,
                              self.dataset.training.mapped_triples], 0)
        all_target2degrees2trids = self.make_partitions(all_tris, target2degrees2entids)
        del all_tris
        # self._to_degree_distribution_charts(target2degrees2entids, 'Entities')
        self._to_degree_distribution_charts(all_target2degrees2trids, 'Triples')
        target2degree2tri_idx = self.make_partitions(self.dataset.testing.mapped_triples, target2degrees2entids)
        # target2m2degree_eval = self.get_partition_eval(target2degree2tri_idx)
        # self._to_degree_eval_charts(target2m2degree_eval, target2degrees2entids)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="CP_CPComplEx_TuckER_RotatE_anyburl")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    parser.add_argument('--cali', type=str, default="True")
    parser.add_argument('--partition', type=str, default="rel_mapping")
    args = parser.parse_args()
    args.models = args.models.split('_')
    anlz = EntDegreeChart(args)
    anlz.analyze()
