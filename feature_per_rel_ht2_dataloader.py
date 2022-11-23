import pandas as pd
from pykeen.evaluation.evaluator import create_sparse_positive_filter_, filter_scores_
from pykeen.typing import MappedTriples, COLUMN_HEAD, COLUMN_TAIL
import torch
from collections import Counter
from FusionDataset import FusionDataset


class PerRelNoSignalDataset(FusionDataset):
    #   combine head and tail scores in one example:
    #   [m1_h_eval, m1_t_eval, ..., m1_h_score, m1_t_score, ... ]
    def __init__(self, mapped_triples: MappedTriples, context_resource, all_pos_triples, num_neg=4):
        super().__init__(mapped_triples, context_resource, all_pos_triples, num_neg)


    def _get_neg_scores_for_training(self, dev_predictions, times=4):
        # Create filter
        targets = [COLUMN_HEAD, COLUMN_TAIL]
        neg_scores = []
        neg_index = []
        for index in range(2):
            # exclude positive triples
            positive_filter, _ = create_sparse_positive_filter_(
                hrt_batch=self.mapped_triples,
                all_pos_triples=self.all_pos_triples,
                relation_filter=None,
                filter_col=targets[index],
            )
            scores = filter_scores_(scores=dev_predictions[index], filter_batch=positive_filter)
            # random pick top_k negs, if no more than top_k, then fill with -999.
            # However the top_k from different model is not always the same
            # Our solution is that we pick the top_k * 2 candidates, and pick the most frequent index
            select_range = self.num_neg * times if self.num_neg * times < scores.shape[-1] else scores.shape[-1]
            scores_k, indices_k = torch.nan_to_num(scores, nan=-999.).topk(k=select_range)
            # remove -999 from scores_k
            nan_index = torch.nonzero(scores_k == -999.)
            indices_k[nan_index[:, 0], nan_index[:, 1]] = int(-1)
            neg_scores.append(scores)
            neg_index.append(indices_k)
        return neg_scores, neg_index

    def _get_rel_eval_scores(self, model_rel_eval):
        rel_mapped = pd.DataFrame(data=self.mapped_triples.numpy()[:, 1], columns=['r'])
        rel_idx = rel_mapped.applymap(lambda x: self.releval2idx[x] if x in self.releval2idx else -1)
        rel_h_t_eval = model_rel_eval[rel_idx.to_numpy().T]
        return rel_h_t_eval

    def generate_training_input_feature(self):
        #   [m1_h_eval, m1_t_eval, ..., m1_h_score, m1_t_score, ... ]
        head_eval = []
        tail_eval = []
        scores_neg = []
        scores_pos = []
        neg_index_topk_times = []
        rel_eval = []
        targets = [COLUMN_HEAD, COLUMN_TAIL]
        for m in self.context_resource['models']:
            m_context = self.context_resource[m]
            m_dev_preds = m_context['eval']
            m_dev_preds = torch.chunk(m_dev_preds, 2, 1)

            m_rel_h_t_eval = self._get_rel_eval_scores(m_context['rel_eval'])
            m_h_eval, m_t_eval = torch.chunk(m_rel_h_t_eval, 2, 1)
            head_eval.append(m_h_eval)
            tail_eval.append(m_t_eval)
            rel_eval.append(m_rel_h_t_eval.flatten())
            # get pos scores
            m_pos_scores = m_dev_preds[0]
            m_pos_scores = m_pos_scores[torch.arange(0, self.mapped_triples.shape[0]),
                                                      self.mapped_triples[:, targets[0]]]
            # [h1,h2...]
            scores_pos.append(m_pos_scores.flatten())  # [s1,s2,...]
            # tri get neg scores
            m_neg_scores, m_neg_index_topk4 = self._get_neg_scores_for_training(m_dev_preds) # [[h1 * candidate, h2 * candicate...][t1,t2...]]
            scores_neg.append(torch.stack(m_neg_scores, 1))  # [h1* candi,h2 * candi...,t1 * candi, t2* candi...]
            neg_index_topk_times.append(torch.stack(m_neg_index_topk4, 1))

        # # eval features
        scores_pos = torch.vstack(scores_pos).T  # [[m1_h1,m2_h1,...] ...]
        rel_eval = torch.vstack(rel_eval).T  # [[m1_h1_eval,m2_h1_eval,...][m1_t1_eval,m2_t1_eval,...] ...]
        head_eval = torch.hstack(head_eval)
        tail_eval = torch.hstack(tail_eval)
        # # pos feature [m1_s1,m2_s1,....m1_h1_eval,m2_h1_eval,...m1_t1_eval,m2_t1_eval,...]
        preserve_shape = rel_eval.shape
        pos_features = torch.concat([rel_eval.reshape(int(preserve_shape[0] / 2), preserve_shape[1] * 2),
                                     scores_pos], 1)
        # neg feature
        neg_features = self._get_multi_model_neg_topk(scores_neg, neg_index_topk_times, [head_eval, tail_eval])
        return pos_features, neg_features

    def _get_multi_model_neg_topk(self, neg_score_ht, neg_index_topk, h_t_rel_eval:[]):
        num_models = len(neg_score_ht)
        selected_scores = []
        neg_scores = torch.stack(neg_score_ht, 0)
        neg_scores = torch.chunk(neg_scores, 2, 2)  # (h, t) tuple
        neg_index = torch.stack(neg_index_topk, 0)
        neg_index = torch.chunk(neg_index, 2, 2)
        ht_eval = torch.cat(h_t_rel_eval, 1)
        #   [e1,h,e3,..],r,t
        #   h,r,[t,e2,...]
        #   The neg examples contain prediction scores and head/tail eval scores
        # we count the most frequent top_k examples from head and tail preds
        for target in range(2):
            target_neg_scores = neg_scores[target].squeeze().transpose(0,1).transpose(1,2)
            topk_idx = neg_index[target].squeeze()
            # count top_k frequent index
            backup_shape = topk_idx.shape
            topk_idx = topk_idx.transpose(0,1).reshape([backup_shape[1], backup_shape[0]*backup_shape[2]])
            idx_df = pd.DataFrame(data=topk_idx.numpy().T)
            idx_df = idx_df.groupby(idx_df.columns, axis=1).apply(lambda x: x.values).apply(lambda y:y.flatten())
            idx_df = idx_df.apply(lambda x:x[x != -1]).apply(lambda x: [a[0] for a in Counter(x).most_common(self.num_neg)])
            tmp_topk_idx = list(idx_df.values)
            fill_topk = []
            for i in tmp_topk_idx:
                if len(i) == self.num_neg:
                    fill_topk.append(i)
                else:
                    i.extend(list(range(self.num_neg - len(i))))
                    fill_topk.append(i)
            fill_topk = torch.as_tensor(fill_topk)
            target_neg_scores = target_neg_scores[torch.arange(0, target_neg_scores.shape[0]).unsqueeze(1), fill_topk]
            selected_scores.append(target_neg_scores)

        eval_repeat = torch.reshape(ht_eval.repeat(1, self.num_neg), (ht_eval.shape[0], self.num_neg, num_models * 2))
        scores_repeat = torch.cat(selected_scores, 1)
        # random select from  top_k * times
        sample_weight = torch.ones(scores_repeat.shape[0], scores_repeat.shape[1])
        sample_index = torch.multinomial(sample_weight, self.num_neg)
        scores_repeat = scores_repeat[torch.arange(0, scores_repeat.shape[0]).unsqueeze(1), sample_index]
        selected_feature = torch.cat([eval_repeat, scores_repeat], 2)
        selected_feature = selected_feature.reshape(selected_feature.shape[0] * selected_feature.shape[1], selected_feature.shape[2])
        selected_feature = torch.nan_to_num(selected_feature, -999.)
        return selected_feature

    def generate_pred_input_feature(self):
        # s1,m1_h_eval,... m1_t_eval
        head_scores = []
        tail_scores = []
        head_eval = []
        tail_eval = []
        num_models = len(self.context_resource['models'])
        for m in self.context_resource['models']:
            m_context = self.context_resource[m]
            m_preds = m_context['preds']
            m_h_preds, m_t_preds = torch.chunk(m_preds, 2, 1)
            head_scores.append(torch.flatten(m_h_preds, start_dim=0, end_dim=1))
            tail_scores.append(torch.flatten(m_t_preds, start_dim=0, end_dim=1))
            m_rel_h_t_eval = self._get_rel_eval_scores(m_context['rel_eval'])
            m_h_eval, m_t_eval = torch.chunk(m_rel_h_t_eval, 2, 1)
            head_eval.append(m_h_eval)
            tail_eval.append(m_t_eval)

        candidate_num = int(head_scores[0].shape[0] / self.mapped_triples.shape[0])
        head_eval = torch.hstack(head_eval).repeat((1, candidate_num)).reshape([candidate_num * self.mapped_triples.shape[0], num_models])
        tail_eval = torch.hstack(tail_eval).repeat((1, candidate_num)).reshape([candidate_num * self.mapped_triples.shape[0], num_models])
        eval_feature = torch.cat([head_eval, tail_eval], 1)
        head_scores = torch.vstack(head_scores).T
        tail_scores = torch.vstack(tail_scores).T
        h_features = torch.concat([eval_feature, head_scores], 1)
        t_features = torch.concat([eval_feature, tail_scores], 1)
        h_t_features = torch.concat([h_features, t_features], 0)
        return h_t_features