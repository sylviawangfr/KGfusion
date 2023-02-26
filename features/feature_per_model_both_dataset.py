import pandas as pd
from pykeen.evaluation.evaluator import create_sparse_positive_filter_, filter_scores_
from pykeen.typing import MappedTriples, COLUMN_HEAD, COLUMN_TAIL
import torch
from features.FusionDataset import FusionDataset, get_multi_model_neg_topk


class PerModelBothDataset(FusionDataset):
    #   combine head and tail scores in one example:
    #   [m1_h_eval, m1_t_eval, ..., m1_h_score, m1_t_score, ... ]
    def __init__(self, mapped_triples: MappedTriples, context_resource, all_pos_triples, num_neg=4):
        super().__init__(mapped_triples, context_resource, all_pos_triples, num_neg)
        self.dim = len(context_resource['models']) * 2


    def generate_training_input_feature(self):
        #   [m1_score, m2_score, ... ]
        scores_neg = []
        scores_pos = []
        both_eval = []
        neg_index_topk_times = []
        for m in self.context_resource['models']:
            m_context = self.context_resource[m]
            # get pos scores
            m_pos_scores = m_context['eval_pos_scores']
            scores_pos.append(m_pos_scores)  # [s1,s2,...]
            # tri get neg scores
            m_neg_scores = m_context['eval_neg_scores']
            m_neg_index_topk4 = m_context['eval_neg_index']
            scores_neg.append(m_neg_scores)  # [h1* candi,h2 * candi...,t1 * candi, t2* candi...]
            neg_index_topk_times.append(m_neg_index_topk4)
            # model eval
            m_hrb_eval = m_context['total_eval']
            m_b_eval = m_hrb_eval[2, 3]
            both_eval.append(m_b_eval.repeat(scores_pos.shape[0], 0))
        # # pos feature [m1_eval, m2_eva., ... m1_s1,m2_s1,....]
        scores_pos = torch.vstack(scores_pos).T
        # neg feature
        scores_neg = get_multi_model_neg_topk(scores_neg, neg_index_topk_times, self.num_neg)
        # eval scores
        both_eval = torch.hstack(both_eval)
        pos_features = torch.concat([both_eval, scores_pos], 1)
        num_models = len(self.context_resource['models'])
        eval_repeat = torch.reshape(both_eval.repeat(1, self.num_neg), (both_eval.shape[0] * self.num_neg, num_models))
        neg_features = torch.cat([eval_repeat, scores_neg], -1)
        return pos_features, neg_features

    def generate_pred_input_feature(self):
        # s1,m1_h_eval,... m1_t_eval
        head_scores = []
        tail_scores = []
        both_eval = []
        num_models = len(self.context_resource['models'])
        for m in self.context_resource['models']:
            m_context = self.context_resource[m]
            m_preds = m_context['preds']
            m_h_preds, m_t_preds = torch.chunk(m_preds, 2, 1)
            head_scores.append(torch.flatten(m_h_preds, start_dim=0, end_dim=1))
            tail_scores.append(torch.flatten(m_t_preds, start_dim=0, end_dim=1))
            m_b_eval = m_context['total_eval'][2, 3]
            both_eval.append(m_b_eval)

        candidate_num = int(head_scores[0].shape[0] / self.mapped_triples.shape[0])
        both_eval = torch.as_tensor(both_eval).repeat(self.mapped_triples.shape[0], 1)
        eval_feature = both_eval.repeat((1, candidate_num)).reshape([candidate_num * self.mapped_triples.shape[0], num_models])
        head_scores = torch.vstack(head_scores).T
        tail_scores = torch.vstack(tail_scores).T
        h_features = torch.concat([eval_feature, head_scores], 1)
        t_features = torch.concat([eval_feature, tail_scores], 1)
        h_t_features = torch.concat([h_features, t_features], 0)
        return h_t_features