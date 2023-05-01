from pykeen.datasets import get_dataset
from tabulate import tabulate
import common_utils
from analysis.group_eval_utils import get_all_pos_triples, group_rank_eval
from context_load_and_run import ContextLoader
import torch
import argparse


class ModelChart():
    def __init__(self, params):
        self.params = params
        self.dataset = get_dataset(
            dataset=params.dataset
        )
        self.context_loader = ContextLoader(self.params.work_dir, self.params.models)
        self.all_pos_triples = get_all_pos_triples(self.dataset)

    def evaluate_preds(self):
        tri_idx = torch.arange(0, self.dataset.testing.mapped_triples.shape[0])
        model_evals = {}
        for m in self.params.models:
            m_context = self.context_loader.load_preds([m], calibrated=self.params.calibrated)
            model2eval = group_rank_eval(self.dataset.testing.mapped_triples,
                                         {m: tri_idx},
                                         m_context[m]['preds'],
                                         self.all_pos_triples)
            model_evals.update(model2eval)
        return model_evals

    def _to_table(self, model_evals):
        header = ['', 'Head \nhit@1','Head \nhit@3','Head \nhit@10', 'Head \nMRR',
                  'Tail \nhit@1', 'Tail \nhit@3','Tail \nhit@10','Tail \n MRR',
                  'Both \nHit@1', 'Both \nHit@3', 'Both \nHit@10', 'Both \n MRR']
        data = []
        for m in self.params.models:
            m_data = [m]
            m_data.extend(model_evals[m].flatten().tolist())
            data.append(m_data)
        table_simple = tabulate(data, header, tablefmt="simple_grid", numalign="center", floatfmt=".3f")
        table_latex = tabulate(data, header, tablefmt="latex", numalign="center", floatfmt=".3f")
        print(table_simple)
        common_utils.save_to_file(table_simple, self.params.work_dir + f'figs/models_eval.txt', mode='w')
        common_utils.save_to_file(table_latex, self.params.work_dir + f'figs/models_eval.txt', mode='a')

    def analyze_test(self):
        model2eval = self.evaluate_preds()
        self._to_table(model2eval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="CP_ComplEx_TuckER_RotatE_anyburl")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    parser.add_argument('--calibrated', type=str, default="True")
    args = parser.parse_args()
    args.models = args.models.split('_')
    tmp = ModelChart(args)
    tmp.analyze_test()
