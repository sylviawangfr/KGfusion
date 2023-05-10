import pandas as pd
from pykeen.evaluation.evaluator import create_sparse_positive_filter_, filter_scores_
from pykeen.typing import MappedTriples, COLUMN_HEAD, COLUMN_TAIL
import torch
from features.FusionDataset import FusionDataset, get_multi_model_neg_topk


class PerRelBestDataset(FusionDataset):
    #   combine head and tail scores in one example:
    #   [m1_h_eval, m1_t_eval, ..., m1_h_score, m1_t_score, ... ]
    def __init__(self, mapped_triples: MappedTriples, context_loader, all_pos_triples, feature='rel',num_neg=4, calibrated=False):
        super().__init__(mapped_triples, context_loader, all_pos_triples, num_neg)
        self.dim = len(context_loader.models) * 2
        self.feature = feature
        self.calibrated = calibrated

    def _get_rel_eval_scores(self, model_rel_eval, rel2idx):
        rel_mapped = pd.DataFrame(data=self.mapped_triples.numpy()[:, 1], columns=['r'])
        rel_idx = rel_mapped.applymap(lambda x: rel2idx[x])
        rel_h_eval = model_rel_eval[rel_idx.to_numpy(), 0, -1]  # the model_rel_eval is in shape [group_num, 3, 4], hit@1,3,10, mrr
        rel_t_eval = model_rel_eval[rel_idx.to_numpy(), 1, -1]  # the model_rel_eval is in shape [group_num, 3, 4], hit@1,3,10, mrr
        return rel_h_eval, rel_t_eval

    def generate_training_input_feature(self):
        pass

    def generate_pred_input_feature(self):
        # s1,m1_h_eval,... m1_t_eval
        head_scores = []
        tail_scores = []
        h_rel_eval = []
        t_rel_eval = []

        num_models = len(self.context_loader.models)
        if self.feature == 'rel':
            m_rel_eval = self.context_loader.load_rel_eval(self.context_loader.models, cache=False)
        else:
            m_rel_eval = self.context_loader.load_rel_mapping_eval(self.context_loader.models, cache=False)
        for m in self.context_loader.models:
            m_preds = self.context_loader.load_preds([m], calibrated=self.calibrated)
            m_h_preds, m_t_preds = torch.chunk(m_preds[m]['preds'], 2, 1)
            head_scores.append(torch.flatten(m_h_preds, start_dim=0, end_dim=1))
            tail_scores.append(torch.flatten(m_t_preds, start_dim=0, end_dim=1))
            m_h_eval, m_t_eval = self._get_rel_eval_scores(m_rel_eval[m]['rel_eval'], m_rel_eval['rel2eval_idx'])
            h_rel_eval.append(m_h_eval)
            t_rel_eval.append(m_t_eval)

        candidate_num = int(head_scores[0].shape[0] / self.mapped_triples.shape[0])
        # assign lower eval scores to zero, so that we pick the best performed model
        rel_h_eval_feature = torch.hstack(h_rel_eval)
        max_h = torch.max(rel_h_eval_feature, dim=1).values.unsqueeze(1)
        max_h_index = torch.nonzero(rel_h_eval_feature != max_h, as_tuple=True)
        rel_h_eval_feature[max_h_index] = 0
        rel_h_eval_feature = rel_h_eval_feature.repeat((1, candidate_num)).reshape([candidate_num * self.mapped_triples.shape[0], num_models])
        rel_t_eval_feature = torch.hstack(t_rel_eval)
        max_t = torch.max(rel_t_eval_feature, dim=1).values.unsqueeze(1)
        max_t_index = torch.nonzero(rel_t_eval_feature != max_t, as_tuple=True)
        rel_t_eval_feature[max_t_index] = 0
        rel_t_eval_feature = rel_t_eval_feature.repeat((1, candidate_num)).reshape([candidate_num * self.mapped_triples.shape[0], num_models])
        head_scores = torch.vstack(head_scores).T
        tail_scores = torch.vstack(tail_scores).T
        h_features = torch.concat([rel_h_eval_feature, head_scores], 1)
        t_features = torch.concat([rel_t_eval_feature, tail_scores], 1)
        h_t_features = torch.concat([h_features, t_features], 0)
        return h_t_features
