import argparse
import numpy as np
import pandas as pd
import torch
from tabulate import tabulate
import common_utils
from analysis.group_eval_utils import group_rank_eval, AnalysisChart, find_relation_mappings
import matplotlib.pyplot as plt


class RelChart(AnalysisChart):
    def __init__(self, params):
        super().__init__(params)

    def make_triple_partitions(self, mapped_triples):
        triples_df = pd.DataFrame(data=mapped_triples.numpy(), columns=['h', 'r', 't'])
        original_groups = triples_df.groupby('r', group_keys=True, as_index=False)
        group_keys = original_groups.groups.keys()
        key2tri_ids = dict()
        for key in group_keys:
            g = original_groups.get_group(key)
            g_index = torch.from_numpy(g.index.values)
            key2tri_ids.update({key: g_index})
        return key2tri_ids

    def get_partition_valid_eval_per_model(self, key2tri_ids):
        m_dict = dict()
        for m in self.params.models:
            m_context = self.context_loader.load_valid_preds([m], cache=False)
            key2eval = group_rank_eval(self.dataset.validation.mapped_triples,
                                       key2tri_ids,
                                       m_context[m]['valid_preds'],
                                       self.all_pos_triples)
            m_dict.update({m: key2eval})
        return m_dict

    def get_partition_test_eval_per_model(self, r2tri_ids):
        m_dict = dict()
        for m in self.params.models:
            m_context = self.context_loader.load_preds([m], cache=False)
            key2eval = group_rank_eval(self.dataset.testing.mapped_triples,
                                       r2tri_ids,
                                       m_context[m]['preds'],
                                       self.all_pos_triples)
            m_dict.update({m: key2eval})
        return m_dict

    def partition_eval_and_save(self, r2tri_ids):
        dir_name = self.params.work_dir + 'rel_eval/'
        common_utils.init_dir(dir_name)
        m2rel2eval_scores = self.get_partition_valid_eval_per_model(r2tri_ids)
        all_relids = self.dataset.relation_to_id.values()
        for model_name in self.params.models:
            rel2head_tail_eval = m2rel2eval_scores[model_name]
            m_head_tail_eval = []
            for r in all_relids:
                r_eval = rel2head_tail_eval[r] if r in rel2head_tail_eval else torch.zeros((3, 4))
                m_head_tail_eval.append(r_eval)
            torch.save(torch.stack(m_head_tail_eval, 0), dir_name + f"{model_name}_rel_eval.pt")

        rel2evalidx = {int(rel): idx for idx, rel in enumerate(all_relids)}
        common_utils.save2json(rel2evalidx, dir_name + f"rel2eval_idx.json")

    def analyze_test(self):
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="CP_CPComplEx_TuckER_RotatE_anyburl")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    parser.add_argument('--cali', type=str, default="True")
    args = parser.parse_args()
    args.models = args.models.split('_')
    tmp = RelChart(args)
    tmp.analyze_test()
