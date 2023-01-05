import pandas as pd
from pykeen.evaluation.evaluator import create_sparse_positive_filter_, filter_scores_
from pykeen.typing import MappedTriples, COLUMN_HEAD, COLUMN_TAIL
import torch
from features.FusionDataset import FusionDataset, get_multi_model_neg_topk


class PerRelEntDataset(FusionDataset):
    #   combine head and tail scores in one example:
    #   [m1_h_eval, m1_t_eval, ..., m1_h_score, m1_t_score, ... ]
    def __init__(self, mapped_triples: MappedTriples, context_resource, all_pos_triples, eval_feature=0, num_neg=4):
        super().__init__(mapped_triples, context_resource, all_pos_triples, num_neg)
        self.dim = len(context_resource['models']) * 2
        self.eval_feature = eval_feature

    def _get_rel_eval_scores(self, model_rel_eval, rel2idx):
        rel_mapped = pd.DataFrame(data=self.mapped_triples.numpy()[:, 1], columns=['r'])
        rel_idx = rel_mapped.applymap(lambda x: rel2idx[x] if x in rel2idx else -1)
        rel_h_t_eval = model_rel_eval[rel_idx.to_numpy().T]
        return rel_h_t_eval

    def _get_h_eval_scores(self, model_h_ent_eval, h_ent2idx):
        head_ent_mapped = pd.DataFrame(data=self.mapped_triples.numpy()[:, 0], columns=['h'])
        h_ent_idx = head_ent_mapped.applymap(lambda x: h_ent2idx[x] if x in h_ent2idx else -1)
        h_ent_eval = model_h_ent_eval[h_ent_idx.to_numpy().T]
        return h_ent_eval

    def _get_t_eval_scores(self, model_t_ent_eval, t_ent2idx):
        tail_ent_mapped = pd.DataFrame(data=self.mapped_triples.numpy()[:, 2], columns=['t'])
        t_ent_idx = tail_ent_mapped.applymap(lambda x: t_ent2idx[x] if x in t_ent2idx else -1)
        t_ent_eval = model_t_ent_eval[t_ent_idx.to_numpy().T]
        return t_ent_eval

    def generate_training_input_feature(self):
        pass
        #   [m1_score, m2_score, ... ]
        # scores_neg = []
        # scores_pos = []
        # h_rel_eval = []
        # t_rel_eval = []
        # h_ent_eval = []
        # t_ent_eval = []
        # neg_index_topk_times = []
        # rel2idx = self.context_resource['releval2idx']
        # h_ent2idx = self.context_resource['h_ent2idx']
        # t_ent2idx = self.context_resource['t_ent2idx']
        # for m in self.context_resource['models']:
        #     m_context = self.context_resource[m]
        #     # get pos scores
        #     m_pos_scores = m_context['eval_pos_scores']
        #     scores_pos.append(m_pos_scores)  # [s1,s2,...]
        #     # tri get neg scores
        #     m_neg_scores = m_context['eval_neg_scores']
        #     m_neg_index_topk4 = m_context['eval_neg_index']
        #     scores_neg.append(m_neg_scores)  # [h1* candi,h2 * candi...,t1 * candi, t2* candi...]
        #     neg_index_topk_times.append(m_neg_index_topk4)
        #     # model eval hit@N
        #     m_hrb_eval = self._get_rel_eval_scores(m_context['rel_eval'], rel2idx)
        #     m_h_eval, m_t_eval, m_b_eval = torch.chunk(m_hrb_eval, 3, 1)
        #     h_rel_eval.append(m_h_eval)
        #     t_rel_eval.append(m_t_eval)
        #     h_ent_eval.append(self._get_h_eval_scores(m_context['h_ent_eval'], h_ent2idx))
        #     t_ent_eval.append(self._get_t_eval_scores(m_context['t_ent_eval'], t_ent2idx))
        # # # pos feature [m1_eval, m2_eva., ... m1_s1,m2_s1,....]
        # scores_pos = torch.vstack(scores_pos).T
        # # neg feature
        # scores_neg = get_multi_model_neg_topk(scores_neg, neg_index_topk_times, self.num_neg)
        # # eval scores
        # _eval = torch.hstack(both_eval)
        # pos_features = torch.concat([both_eval, scores_pos], 1)
        # num_models = len(self.context_resource['models'])
        # eval_repeat = torch.reshape(both_eval.repeat(1, self.num_neg), (both_eval.shape[0] * self.num_neg, num_models))
        # neg_features = torch.cat([eval_repeat, scores_neg], -1)
        # return pos_features, neg_features

    def generate_pred_input_feature(self):
        # s1,m1_h_eval,... m1_t_eval
        head_scores = []
        tail_scores = []
        h_rel_eval = []
        t_rel_eval = []
        h_ent_eval = []
        t_ent_eval = []
        rel2idx = self.context_resource['releval2idx']
        h_ent2idx = self.context_resource['h_ent2idx']
        t_ent2idx = self.context_resource['t_ent2idx']
        num_models = len(self.context_resource['models'])
        for m in self.context_resource['models']:
            m_context = self.context_resource[m]
            m_preds = m_context['preds']
            m_h_preds, m_t_preds = torch.chunk(m_preds, 2, 1)
            head_scores.append(torch.flatten(m_h_preds, start_dim=0, end_dim=1))
            tail_scores.append(torch.flatten(m_t_preds, start_dim=0, end_dim=1))
            m_rel_h_t_eval = self._get_rel_eval_scores(m_context['rel_eval'], rel2idx)
            m_h_eval, m_t_eval, m_b_eval = torch.chunk(m_rel_h_t_eval, 3, 1)
            h_rel_eval.append(m_h_eval)
            t_rel_eval.append(m_t_eval)
            h_ent_eval.append(self._get_h_eval_scores(m_context['h_ent_eval'], h_ent2idx))  # tail predicition
            t_ent_eval.append(self._get_t_eval_scores(m_context['t_ent_eval'], t_ent2idx))  # head predicition

        candidate_num = int(head_scores[0].shape[0] / self.mapped_triples.shape[0])
        rel_h_eval_feature = torch.hstack(h_rel_eval).repeat((1, candidate_num)).reshape([candidate_num * self.mapped_triples.shape[0], num_models])
        rel_t_eval_feature = torch.hstack(t_rel_eval).repeat((1, candidate_num)).reshape([candidate_num * self.mapped_triples.shape[0], num_models])
        #   <?, r, t>
        ent_h_eval_feature = torch.vstack(t_ent_eval).T.repeat((1, candidate_num)).reshape([candidate_num * self.mapped_triples.shape[0], num_models])
        #   <h, r, ?>
        ent_t_eval_feature = torch.vstack(h_ent_eval).T.repeat((1, candidate_num)).reshape([candidate_num * self.mapped_triples.shape[0], num_models])
        head_scores = torch.vstack(head_scores).T
        tail_scores = torch.vstack(tail_scores).T
        h_features = torch.concat([rel_h_eval_feature, ent_h_eval_feature, head_scores], 1)
        t_features = torch.concat([rel_t_eval_feature, ent_t_eval_feature, tail_scores], 1)
        h_t_features = torch.concat([h_features, t_features], 0)
        return h_t_features