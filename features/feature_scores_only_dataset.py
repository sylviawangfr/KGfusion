import random

import pandas as pd
from pykeen.evaluation.evaluator import create_sparse_positive_filter_, filter_scores_
from pykeen.typing import MappedTriples, COLUMN_HEAD, COLUMN_TAIL
import torch
from collections import Counter

from context_load_and_run import ContextLoader
from features.FusionDataset import FusionDataset, padding_sampling, get_multi_model_neg_topk
from common_utils import chart_input


class ScoresOnlyDataset(FusionDataset):
    #   combine head and tail scores in one example:
    #   [m1_h_eval, m1_t_eval, ..., m1_h_score, m1_t_score, ... ]
    def __init__(self, mapped_triples: MappedTriples, context_loader: ContextLoader, all_pos_triples, num_neg=4, calibrated=False, models=[]):
        super().__init__(mapped_triples, context_loader, all_pos_triples, num_neg)
        if len(models) == 0:
            self.models = context_loader.models
        else:
            self.models = models
        self.dim = len(self.models)
        self.calibrated = calibrated

    def generate_training_input_feature(self):
        #   [m1_score, m2_score, ... ]
        scores_neg = []
        scores_pos = []
        neg_index_topk_times = []
        for m in self.models:
            m_context = self.context_loader.load_eval_examples([m], cache=False)
            # get pos scores
            m_pos_scores = m_context[m]['eval_pos_scores']
            scores_pos.append(m_pos_scores)  # [s1,s2,...]
            # tri get neg scores
            m_neg_scores = m_context[m]['eval_neg_scores']
            m_neg_index_topk4 = m_context[m]['eval_neg_index']
            scores_neg.append(m_neg_scores)  #
            neg_index_topk_times.append(m_neg_index_topk4)

        # # pos feature [m1_s1,m2_s1,....]
        pos_features = torch.vstack(scores_pos).T
        # neg feature
        neg_features = get_multi_model_neg_topk(scores_neg, neg_index_topk_times, num_neg=self.num_neg)
        # debug
        # p1 = torch.ones(pos_features.shape[0])
        # p2 = torch.ones(neg_features.shape[0])
        # chart_input(pos_features[p1.multinomial(150)], neg_features[p2.multinomial(300)], 'feature.png')
        return pos_features, neg_features

    def generate_pred_input_feature(self):
        # s1,s2,...
        head_scores = []
        tail_scores = []
        for m in self.models:
            m_context = self.context_loader.load_preds([m], calibrated=self.calibrated)
            m_preds = m_context[m]['preds']
            m_h_preds, m_t_preds = torch.chunk(m_preds, 2, 1)
            head_scores.append(torch.flatten(m_h_preds, start_dim=0, end_dim=1))
            tail_scores.append(torch.flatten(m_t_preds, start_dim=0, end_dim=1))

        head_scores = torch.vstack(head_scores).T
        tail_scores = torch.vstack(tail_scores).T
        h_t_features = torch.concat([head_scores, tail_scores], 0)
        h_t_features = h_t_features.type(torch.float32)
        return h_t_features


