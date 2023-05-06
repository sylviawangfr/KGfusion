import argparse
import pandas as pd
import torch
from tabulate import tabulate

import common_utils
from analysis.group_eval_utils import group_rank_eval, AnalysisChart


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
            m_context = self.context_loader.load_preds([m], calibrated=True)
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

    def _to_table(self, m2rel2eval, key2tri_ids, all_key2tri_ids):
        rel2count = {k: key2tri_ids[k].shape[0] for k in key2tri_ids}
        rel2count_all = {k: all_key2tri_ids[k].shape[0] for k in all_key2tri_ids}
        rel2count = dict(sorted(rel2count.items(), key=lambda item: item[1], reverse=True))
        rel2count_all = dict(sorted(rel2count_all.items(), key=lambda item: item[1], reverse=True))
        header = ['', 'num_train', 'num_test']
        for m in self.params.models:
            header.append(m)
        col_num = 3 + len(self.params.models)
        row_num = len(rel2count.keys())
        data = [[0 for j in range(col_num)] for i in range(row_num)]
        relid2name = {v: k for k, v in self.dataset.relation_to_id.items()}
        for i, rel in enumerate(rel2count.keys()):
            data[i][0] = relid2name[rel]
            data[i][1] = rel2count_all[rel]
            data[i][2] = rel2count[rel]
        for j, m in enumerate(self.params.models):
            rel2eval = m2rel2eval[m]
            for i, rel in enumerate(rel2count.keys()):
                data[i][j + 3] = rel2eval[rel][2, -1].item()

        table_simple = tabulate(data, header, tablefmt="simple_grid", numalign="center", floatfmt=".3f")
        table_latex = tabulate(data, header, tablefmt="latex", numalign="center", floatfmt=".3f")
        print(table_simple)
        common_utils.save_to_file(table_simple, self.params.work_dir + f'figs/rel_eval.txt', mode='w')
        common_utils.save_to_file(table_latex, self.params.work_dir + f'figs/rel_eval.txt', mode='a')

    def analyze_test(self):
        all_tris = torch.cat([self.dataset.testing.mapped_triples,
                              self.dataset.validation.mapped_triples,
                              self.dataset.training.mapped_triples], 0)
        all_key2tri_ids = self.make_triple_partitions(all_tris)
        del all_tris
        key2tri_ids = self.make_triple_partitions(self.dataset.testing.mapped_triples)
        m2rel2eval = self.get_partition_test_eval_per_model(key2tri_ids)
        self._to_table(m2rel2eval, key2tri_ids, all_key2tri_ids)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="CP_ComplEx_TuckER_RotatE_anyburl")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    parser.add_argument('--cali', type=str, default="True")
    args = parser.parse_args()
    args.models = args.models.split('_')
    tmp = RelChart(args)
    tmp.analyze_test()
