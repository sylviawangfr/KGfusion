import argparse
import numpy as np
import pandas as pd
import torch
from tabulate import tabulate
import common_utils
from analysis.group_eval_utils import group_rank_eval, AnalysisChart, find_relation_mappings
import matplotlib.pyplot as plt


class RelMappingChart(AnalysisChart):
    def __init__(self, params):
        super().__init__(params)
        self.mappings = ['1-1', '1-n', 'n-1', 'n-m']
        self.rel_mappings = find_relation_mappings(self.dataset)

    def make_triple_partitions(self, mapped_triples):
        triples_df = pd.DataFrame(data=mapped_triples.numpy(), columns=['h', 'r', 't'], )
        key2tri_ids = dict()
        for key in self.mappings:
            tmp_rels = self.rel_mappings[key]
            tri_group = triples_df.query('r in @tmp_rels')
            if len(tri_group.index) > 0:
                g_index = torch.from_numpy(tri_group.index.values)
                key2tri_ids.update({key: g_index})
            else:
                key2tri_ids.update({key: []})
        return key2tri_ids

    def get_partition_valid_eval_per_model(self, key2tri_ids):
        m_dict = dict()
        for m in self.params.models:
            m_context = self.context_loader.load_valid_preds([m], cache=False)
            key2eval = group_rank_eval(
                self.dataset.validation.mapped_triples,
                key2tri_ids,
                m_context[m]['valid_preds'], self.all_pos_triples)
            m_dict.update({m: key2eval})
        return m_dict

    def get_partition_test_eval_per_model(self, key2tri_ids):
        m_dict = dict()
        for m in self.params.models:
            m_context = self.context_loader.load_preds([m], calibrated=True)
            key2eval = group_rank_eval(
                self.dataset.testing.mapped_triples,
                key2tri_ids,
                m_context[m]['preds'], self.all_pos_triples)
            m_dict.update({m: key2eval})
        return m_dict

    def partition_eval_and_save(self, key2tri_ids):
        dir_name = self.params.work_dir + 'rel_mapping_eval/'
        common_utils.init_dir(dir_name)
        m2mapping2eval_scores = self.get_partition_valid_eval_per_model(key2tri_ids)
        for model_name in self.params.models:
            mapping2head_tail_eval = m2mapping2eval_scores[model_name]
            head_tail_eval = [mapping2head_tail_eval[mapping_type] for mapping_type in self.mappings]
            torch.save(torch.stack(head_tail_eval, 0), dir_name + f"{model_name}_rel_mapping_eval.pt")

        rel2evalidx = dict()
        for idx, mapping_type in enumerate(self.mappings):
            mapped_rels = self.rel_mappings[mapping_type]
            rel2evalidx.update({int(rel): idx for rel in mapped_rels})
        common_utils.save2json(rel2evalidx, dir_name + f"rel2eval_idx.json")

    def _to_bar_chart(self, m2eval, file_name):
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
        # plt.show()
        plt.savefig(self.params.work_dir + f'figs/{file_name}', dpi=600)

    def _to_pie_chart(self, key2tri_ids, title_keyword):
        labels = key2tri_ids.keys()
        values = [len(key2tri_ids[t]) for t in key2tri_ids]
        fig, ax = plt.subplots()
        ax.pie(values, labels=labels, autopct='%1.1f%%', colors=['olivedrab', 'rosybrown', 'gray', 'saddlebrown'])
        ax.set_title(f"{title_keyword} Partitions on Relation Mappings")
        plt.savefig(self.params.work_dir + f'figs/{title_keyword}_rel_mapping_pie.png', dpi=600)

    def _to_table(self, m2eval):
        header = ['', '1-1\nh', '1-1\nt',
                  '1-n\nh', '1-n\nt',
                  'n-1\nh', 'n-1\nt',
                  'n-m\nh', 'n-m\nt']
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
        table_simple = tabulate(data, header, tablefmt="simple_grid", numalign="center", floatfmt=".3f")
        table_latex = tabulate(data, header, tablefmt="latex", numalign="center", floatfmt=".3f")
        common_utils.save_to_file(table_simple, self.params.work_dir + f'figs/rel_mapping_eval.txt', mode='w')
        common_utils.save_to_file(table_latex, self.params.work_dir + f'figs/rel_mapping_eval.txt', mode='a')

    def analyze_test(self):
        common_utils.init_dir(self.params.work_dir + 'figs/')
        all_tris = torch.cat([self.dataset.testing.mapped_triples,
                              self.dataset.validation.mapped_triples,
                              self.dataset.training.mapped_triples], 0)
        all_key2tri_ids = self.make_triple_partitions(all_tris)
        del all_tris
        self._to_pie_chart(all_key2tri_ids, "Dataset")
        key2tri_ids = self.make_triple_partitions(self.dataset.testing.mapped_triples)
        self._to_pie_chart(key2tri_ids, "Testset")
        m2eval = self.get_partition_test_eval_per_model(key2tri_ids)
        self._to_bar_chart(m2eval, "rel_mapping_test_mrr.png")
        self._to_table(m2eval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="CP_CPComplEx_TuckER_RotatE_anyburl")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    parser.add_argument('--cali', type=str, default="True")
    args = parser.parse_args()
    args.models = args.models.split('_')
    tmp = RelMappingChart(args)
    tmp.analyze_test()
