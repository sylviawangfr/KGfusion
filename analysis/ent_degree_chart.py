import argparse
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
from tabulate import tabulate
from itertools import chain
import common_utils
from analysis.group_eval_utils import group_rank_eval, AnalysisChart
import matplotlib.pyplot as plt
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


def cut_tail_and_split(degree2ids, num_patt):
    # cut long tail
    id_count = 0
    tail_degree = 0
    total = sum([len(list(l)) for l in degree2ids.values()])
    for i, ids in degree2ids.items():
        id_count += len(ids)
        if id_count / total >= 0.99:
            tail_degree = i
            break
    range_degree = range(0, tail_degree + 1)
    slots = np.array_split(range_degree, num_patt - 1)
    tail_slot = np.arange(tail_degree, max(list(degree2ids.keys())) + 1)
    slots.append(tail_slot)
    return slots


def make_entity_partitions(self, target2degrees2entids):
    target2p2ids = dict()
    for i, (target, degree2ent_ids) in enumerate(target2degrees2entids.items()):
        x_slots = cut_tail_and_split(degree2ent_ids, self.params.num_p)
        group2entids = dict()
        for slot_degrees in x_slots:
            slot_entids = [degree2ent_ids[d] for d in slot_degrees if d in degree2ent_ids]
            ent_ids = list(chain.from_iterable(slot_entids))
            slot_key = f"{slot_degrees[0]}-{slot_degrees[-1]}"
            group2entids.update({slot_key: ent_ids})
        target2p2ids.update({target: group2entids})
    return target2p2ids


class EntDegreeChart(AnalysisChart):
    def __init__(self, params):
        super().__init__(params)
        self.target2degrees2entids = find_degree_groups(self.dataset)
        self.target2partition2entids = make_entity_partitions(self.target2degrees2entids)

    def make_triple_partitions(self, mapped_triples):
        tri_df = pd.DataFrame(data=mapped_triples, columns=['h', 'r', 't'])
        query_strs = ["h in @ent_ids", "t in @ent_ids", "h in @ent_ids or t in @ent_ids"]
        target2p2tri_idx = dict()
        for i, (target, partition2ent_ids) in enumerate(self.target2partition2entids.items()):
            partition2tri_idx = dict()
            for partition in partition2ent_ids:
                ent_ids = partition2ent_ids[partition]
                tri_group = tri_df.query(query_strs[i])
                if len(tri_group.index) > 0:
                    g_index = tri_group.index.values
                    partition2tri_idx.update({partition: g_index})
                else:
                    partition2tri_idx.update({partition: []})
            target2p2tri_idx.update({target: partition2tri_idx})
        return target2p2tri_idx

    def _padding(self, target2degree2ids):
        padded = {}
        for target, degree2ids in target2degree2ids.items():
            range_degree = range(0, max(list(degree2ids.keys())) + 1)
            padded_degrees2idx = {d : degree2ids[d] if d in degree2ids else [] for d in range_degree}
            padded.update({target: padded_degrees2idx})
        return padded

    def get_partition_test_eval_per_model(self, target2tri_idx):
        target2m2degree_eval = dict()
        for target, partition2tri_idx in target2tri_idx.items():
            m2eval = dict()
            for m in self.params.models:
                m_preds = self.context_loader.load_preds([m], calibrated=False, cache=False)
                m_degree_group_eval = group_rank_eval(self.dataset.testing.mapped_triples,
                                                      partition2tri_idx,
                                                      m_preds[m]['preds'],
                    self.all_pos_triples)
                m2eval.update({m: m_degree_group_eval})
            target2m2degree_eval.update({target: m2eval})
        return target2m2degree_eval

    def get_partition_valid_eval_per_model(self, target2tri_idx):
        target2m2degree_eval = dict()
        for target, partition2tri_idx in target2tri_idx.items():
            m2eval = dict()
            for m in self.params.models:
                m_valid_preds = self.context_loader.load_valid_preds([m], cache=False)
                m_degree_group_eval = group_rank_eval(
                                                      self.dataset.validation.mapped_triples,
                                                      partition2tri_idx,
                                                      m_valid_preds[m]['valid_preds'], self.all_pos_triples)
                m2eval.update({m: m_degree_group_eval})
            target2m2degree_eval.update({target: m2eval})
        return target2m2degree_eval

    def partition_eval_and_save(self, target2partition2tri_idx):
        dir_name = self.params.work_dir + 'degree_eval/'
        common_utils.init_dir(dir_name)
        target2m2partition2eval = self.get_partition_valid_eval_per_model(target2partition2tri_idx)
        for in_or_out in target2m2partition2eval:
            m2partition2eval_scores = target2m2partition2eval[in_or_out]
            for model_name in args.models:
                partition2eval = m2partition2eval_scores[model_name]
                head_tail_eval = [partition2eval[partition_key] for partition_key in partition2eval.keys()]
                torch.save(torch.stack(head_tail_eval, 0), args.work_dir + f"{model_name}_{in_or_out.lower()}_degree_eval.pt")

            partition2ent_idx = self.target2partition2entids[in_or_out]
            entid2evalidx = dict()
            for idx, (partition, entids) in enumerate(partition2ent_idx.items()):
                entid2evalidx.update({int(entid): idx for entid in entids})
            common_utils.save2json(entid2evalidx, dir_name + f"{in_or_out.lower()}_id2eval_idx.json")

    def _to_degree_distribution_charts(self, target2degrees2ids, y_lable='Triple'):
        for target, degree2ids in target2degrees2ids.items():
            degree = range(0, len(degree2ids.keys()))
            value = [len(degree2ids[i]) for i in degree2ids]
            fig, ax = plt.subplots()
            ax.bar(degree, value)
            ax.set_ylabel(f'Number of {y_lable}')
            ax.set_xlabel(f'{target} Degree')
            ax.set_xticks(degree, degree2ids.keys())
            ax.set_title(f'{y_lable} Distribution')
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            plt.savefig(self.params.work_dir + f'figs/{y_lable}_{target}_degree_partition.png', dpi=600)

    def _to_pie_chart(self, target2degree2ids, title_keyword):
        for target, degree2ids in target2degree2ids.items():
            legend = degree2ids.keys()
            values = [len(degree2ids[slot]) for slot in degree2ids]
            fig, ax = plt.subplots()
            ax.pie(values, labels=legend, autopct='%1.1f%%')
            ax.set_title(f"{title_keyword} Partitions on {target}-degree")
            plt.savefig(self.params.work_dir + f'figs/{title_keyword}_{target}_degree_pie.png', dpi=600)

    def _to_degree_eval_charts(self, target2m2degree_eval):
        to_pick_colors = list(mcolors.TABLEAU_COLORS.values())
        colors = {m: to_pick_colors[idx] for idx, m in enumerate(self.params.models)}
        ax_titles = ['Head', 'Tail', 'Both']
        for target, m2degree2eval in target2m2degree_eval.items():
            x = np.arange(0, self.params.num_p)
            fig, axs = plt.subplots(1, 3, sharex=False, sharey=True, figsize=[12, 4])
            # draw lines for all degree set
            width = 0.15  # the width of the bars
            multiplier = 0
            for m, degree2eval in m2degree2eval.items():
                offset = width * multiplier
                for t in range(0, 3):
                    m_data = [degree2eval[i][t][-1] for i in degree2eval]
                    rects = axs[t].bar(x + offset, m_data, width, color=colors[m])
                multiplier += 1
            # Add some text for labels, title and custom x-axis tick labels, etc.
            axs[0].set_ylabel('MRR')
            for idx, ax in enumerate(axs):
                ax.set_xticks(x + width, degree2eval.keys())
                plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
                ax.set_title(ax_titles[idx])
            fig.legend(colors.keys(), loc='upper center', ncol=len(self.params.models))
            plt.subplots_adjust(top=0.85, left=0.08, bottom=0.15, right=0.99)
            plt.savefig(self.params.work_dir + f'figs/{target}_degree_eval.png', dpi=600)
            # plt.show()

    def _to_table(self, target2m2degree_eval):
        for target, m2degree2eval in target2m2degree_eval.items():
            data = []
            header = ['']
            for m in self.params.models:
                m_data = [m]
                m_d2eval = m2degree2eval[m]
                if len(header) == 1:
                    columns = [[f"{d}\nh", f"{d}\nt"] for d in m_d2eval]
                    header.extend(list(chain.from_iterable(columns)))
                for d in m_d2eval:
                    m_data.extend([m_d2eval[d][0, -1].item(), m_d2eval[d][1, -1].item()])
                data.append(m_data)
            table_simple = tabulate(data, header, tablefmt="simple_grid", numalign="center", floatfmt=".3f")
            table_latex = tabulate(data, header, tablefmt="latex", numalign="center", floatfmt=".3f")
            common_utils.save_to_file(table_simple, self.params.work_dir + f'figs/{target}_degree_eval.txt', mode='w')
            common_utils.save_to_file(table_latex, self.params.work_dir + f'figs/{target}_degree_eval.txt', mode='a')

    def analyze_test(self):
        common_utils.init_dir(self.params.work_dir + 'figs/')
        self._to_degree_distribution_charts(self.target2partition2entids, "Entity")
        all_tris = torch.cat([self.dataset.testing.mapped_triples,
                              self.dataset.validation.mapped_triples,
                              self.dataset.training.mapped_triples], 0)
        all_target2partition2trids = self.make_triple_partitions(all_tris)
        del all_tris
        self._to_degree_distribution_charts(all_target2partition2trids, "Triple")
        test_target2partition2tri = self.make_triple_partitions(self.dataset.testing.mapped_triples)
        target2m2degree_eval = self.get_partition_test_eval_per_model(test_target2partition2tri)
        self._to_table(target2m2degree_eval)
        self._to_degree_eval_charts(target2m2degree_eval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="CP_CPComplEx_TuckER_RotatE_anyburl")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    parser.add_argument('--cali', type=str, default="True")
    parser.add_argument('--num_p', type=int, default=10)
    args = parser.parse_args()
    args.models = args.models.split('_')
    anlz = EntDegreeChart(args)
    anlz.analyze_test()
