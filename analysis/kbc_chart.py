from abc import abstractmethod, ABC
import pykeen
from pykeen.datasets import get_dataset
from pykeen.utils import prepare_filter_triples

from analysis.group_eval_utils import get_all_pos_triples, group_rank_eval
from context_load_and_run import ContextLoader
from typing import Optional
from pykeen.constants import TARGET_TO_INDEX
from pykeen.evaluation.evaluator import filter_scores_
from pykeen.evaluation.evaluator import create_sparse_positive_filter_
from pykeen.typing import MappedTriples, Target, LABEL_TAIL, LABEL_HEAD, COLUMN_HEAD, COLUMN_TAIL
from torch import FloatTensor
import torch
import pandas as pd
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

    def analyze_test(self):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="CP_ComplEx_TuckER_RotatE_anyburl")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    parser.add_argument('--cali', type=str, default="True")
    args = parser.parse_args()
    args.models = args.models.split('_')
    tmp = ModelChart(args)
    tmp.evaluate_preds()
