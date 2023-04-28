import pandas as pd
from pykeen.typing import MappedTriples
import torch

from context_load_and_run import ContextLoader
from features.FusionDataset import FusionDataset


class PerEntDegreeDataset(FusionDataset):
    #   combine head and tail scores in one example:
    #   [m1_In_eval,..., m1_Out_eval, ..., m1_h_score,... m1_t_score, ... ]
    def __init__(self, mapped_triples: MappedTriples, context_loader: ContextLoader, all_pos_triples, num_neg=4,
                 calibrated=True):
        super().__init__(mapped_triples, context_loader, all_pos_triples, num_neg)
        self.dim = len(context_loader.models) * 2
        self.calibrated = calibrated

    def _get_out_degree_tail_pred_eval_scores(self, model_h_ent_eval, h_ent2idx):
        # Use out-degree for tail pred
        head_ent_mapped = pd.DataFrame(data=self.mapped_triples.numpy()[:, 0], columns=['h'])
        h_ent_idx = head_ent_mapped.applymap(lambda x: h_ent2idx[x] if x in h_ent2idx else -1)
        h_ent_eval = model_h_ent_eval[h_ent_idx.to_numpy(), 1, -1]  # the tail eval score, the model eval is in shape [group_num, 3, 4], hit@1,3,10, mrr
        return h_ent_eval

    def _get_in_degree_head_pred_eval_scores(self, model_t_ent_eval, t_ent2idx):
        # Use in-degree for head pred
        tail_ent_mapped = pd.DataFrame(data=self.mapped_triples.numpy()[:, 2], columns=['t'])
        t_ent_idx = tail_ent_mapped.applymap(lambda x: t_ent2idx[x] if x in t_ent2idx else -1)
        t_ent_eval = model_t_ent_eval[t_ent_idx.to_numpy(), 0, -1]  # use head eval score
        return t_ent_eval

    def generate_training_input_feature(self):
        pass

    def generate_pred_input_feature(self):
        # s1,m1_h_eval,... m1_t_eval
        head_scores = []
        tail_scores = []
        tail_pred_using_h_ent_eval = []
        head_pred_using_t_ent_eval = []

        num_models = len(self.context_loader.models)
        m_degree_eval = self.context_loader.load_ent_degree_eval(self.context_loader.models, cache=False)
        for m in self.context_loader.models:
            m_preds = self.context_loader.load_preds([m], calibrated=self.calibrated)
            m_h_preds, m_t_preds = torch.chunk(m_preds[m]['preds'], 2, 1)
            head_scores.append(torch.flatten(m_h_preds, start_dim=0, end_dim=1))
            tail_scores.append(torch.flatten(m_t_preds, start_dim=0, end_dim=1))
            head_pred_using_t_ent_eval.append(self._get_in_degree_head_pred_eval_scores(m_degree_eval[m]['in_degree_eval'], m_degree_eval['in_id2eval_idx']))  # head predicition
            tail_pred_using_h_ent_eval.append(self._get_out_degree_tail_pred_eval_scores(m_degree_eval[m]['out_degree_eval'], m_degree_eval['out_id2eval_idx']))  # tail predicition

        candidate_num = int(head_scores[0].shape[0] / self.mapped_triples.shape[0])
        #   <?, r, t>
        ent_h_pred_eval_feature = torch.hstack(head_pred_using_t_ent_eval).repeat((1, candidate_num)).reshape([candidate_num * self.mapped_triples.shape[0], num_models])
        #   <h, r, ?>
        ent_t_pred_eval_feature = torch.hstack(tail_pred_using_h_ent_eval).repeat((1, candidate_num)).reshape([candidate_num * self.mapped_triples.shape[0], num_models])
        head_scores = torch.vstack(head_scores).T
        tail_scores = torch.vstack(tail_scores).T
        h_features = torch.concat([ent_h_pred_eval_feature, head_scores], 1)
        t_features = torch.concat([ent_t_pred_eval_feature, tail_scores], 1)
        h_t_features = torch.concat([h_features, t_features], 0)
        return h_t_features