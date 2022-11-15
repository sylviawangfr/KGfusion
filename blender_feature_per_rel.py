import logging
import os
from typing import Optional
import pandas as pd
from pykeen.evaluation import RankBasedEvaluator
from pykeen.constants import TARGET_TO_INDEX
from pykeen.evaluation.evaluator import Evaluator
from pykeen.evaluation.evaluator import create_sparse_positive_filter_, create_dense_positive_mask_, filter_scores_
from pykeen.typing import MappedTriples, COLUMN_HEAD, COLUMN_TAIL, Target
from pykeen.utils import resolve_device
from torch import Tensor, FloatTensor
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter


def get_neg_scores_for_training(
        triples: MappedTriples,
        predictions,  # [head_pred, tail_pred]
        all_pos_triples: Optional[MappedTriples], top_k=4):
    # Create filter
    targets = [COLUMN_HEAD, COLUMN_TAIL]
    neg_scores = []
    neg_index = []
    for index in range(2):
        # exclude positive triples
        positive_filter, _ = create_sparse_positive_filter_(
            hrt_batch=triples,
            all_pos_triples=all_pos_triples,
            relation_filter=None,
            filter_col=targets[index],
        )
        scores = filter_scores_(scores=predictions[index], filter_batch=positive_filter)
        # random pick top_k negs, if no more than top_k, then fill with -999.
        # However the top_k from different model is not always the same
        # Our solution is that we pick the top_k * 2 candidates, and pick the most frequent index
        scores_k, indices_k = torch.nan_to_num(scores, nan=-999.).topk(k=top_k * 2)
        # remove -999 from scores_k
        nan_index = torch.nonzero(scores_k == -999.)
        indices_k[nan_index[:, 0], nan_index[:, 1]] = int(-1)
        neg_scores.append(scores)
        neg_index.append(indices_k)
        # positive_mask = create_dense_positive_mask_(zero_tensor=torch.zeros_like(scores), filter_batch=positive_filter)
    return neg_scores, neg_index


def get_rel_eval_scores(mapped_triples: MappedTriples, model_rel_eval, releval2idx: dict):
    # triples_df = pd.DataFrame(data=mapped_triples.numpy(), columns=['h', 'r', 't'])
    rel_mapped = pd.DataFrame(data=mapped_triples.numpy()[:, 1], columns=['r'])
    rel_idx = rel_mapped.applymap(lambda x: releval2idx[x] if x in releval2idx else -1)
    # num_triples = rel_mapped.shape[1]
    rel_h_t_eval = model_rel_eval[rel_idx.to_numpy().T]
    return rel_h_t_eval


def generate_training_input_feature(mapped_triples: MappedTriples, context_resource, all_pos_triples, num_neg = 4):
    head_eval = []
    tail_eval = []
    scores_neg = []
    scores_pos = []
    neg_index_topk_double = []
    rel_eval = []
    targets = [COLUMN_HEAD, COLUMN_TAIL]
    for m in context_resource['models']:
        m_context = context_resource[m]
        m_preds = m_context['eval']
        m_preds = torch.chunk(m_preds, 2, 1)
        m_ht_pos_scores = []
        m_rel_h_t_eval = get_rel_eval_scores(mapped_triples, m_context['rel_eval'], context_resource['releval2idx'])
        m_h_eval, m_t_eval = torch.chunk(m_rel_h_t_eval, 2, 1)
        head_eval.append(m_h_eval)
        tail_eval.append(m_t_eval)
        rel_eval.append(m_rel_h_t_eval.flatten())
        # get pos scores
        for idx, target in enumerate(targets):
            m_pos_scores_target = m_preds[idx]
            m_pos_scores_target = m_pos_scores_target[torch.arange(0, mapped_triples.shape[0]), mapped_triples[:, targets[idx]]]
            m_ht_pos_scores.append(m_pos_scores_target)  # [[h1,h2...],[t1,t2...]]

        m_ht_pos_scores = torch.vstack(m_ht_pos_scores).T
        scores_pos.append(m_ht_pos_scores.flatten())  # [h1,t1,h2,t2...]
        # tri get neg scores
        m_neg_scores, m_neg_index_topk2 = get_neg_scores_for_training(mapped_triples, m_preds, all_pos_triples, num_neg)
        scores_neg.append(torch.stack(m_neg_scores, 1))  # [h,t]
        neg_index_topk_double.append(torch.stack(m_neg_index_topk2, 1))
    print("test")
    # # eval features
    scores_pos = torch.vstack(scores_pos).T  # [h1,t1,h2,t2,...]
    rel_eval = torch.vstack(rel_eval).T  # [h1,t1,h2,t2,...]
    head_eval = torch.hstack(head_eval)
    tail_eval = torch.hstack(tail_eval)
    # # pos feature
    zero_one_signal = torch.as_tensor([0, 1]).repeat(mapped_triples.shape[0]).unsqueeze(dim=1)  # [0,1,0,1....]
    pos_features = torch.concat([zero_one_signal, rel_eval, scores_pos], 1)
    # neg feature
    neg_features = get_multi_model_neg_topk(scores_neg, neg_index_topk_double, num_neg, [head_eval, tail_eval])
    return pos_features, neg_features


def get_multi_model_neg_topk(neg_score_ht, neg_index_topk, top_k, h_t_rel_eval:[]):
    num_models = len(neg_score_ht)
    selected_scores = []
    neg_scores = torch.stack(neg_score_ht, 0)
    neg_scores = torch.chunk(neg_scores, 2, 2)  # (h, t) tuple
    neg_index = torch.stack(neg_index_topk, 0)
    neg_index = torch.chunk(neg_index, 2, 2)
    eval_features = []# (h, t) tuple
    for target in range(2):
        target_neg_scores = neg_scores[target].squeeze().transpose(0,1).transpose(1,2)
        topk_idx = neg_index[target].squeeze()
        # count top_k frequent index
        topk_idx = topk_idx.transpose(0,1).reshape([199, 3*8])
        idx_df = pd.DataFrame(data=topk_idx.numpy().T)
        idx_df = idx_df.groupby(idx_df.columns, axis=1).apply(lambda x: x.values).apply(lambda y:y.flatten())
        idx_df = idx_df.apply(lambda x:x[x != -1]).apply(lambda x: [a[0] for a in Counter(x).most_common(top_k)])
        tmp_topk_idx = list(idx_df.values)
        fill_topk = []
        for i in tmp_topk_idx:
            if len(i) == top_k:
                fill_topk.append(i)
            else:
                i.extend(list(range(top_k - len(i))))
                fill_topk.append(i)
        fill_topk = torch.as_tensor(fill_topk)
        target_neg_scores = target_neg_scores[torch.arange(0, target_neg_scores.shape[0]).unsqueeze(1), fill_topk]
        selected_scores.append(target_neg_scores)
        target_eval = torch.reshape(h_t_rel_eval[target].repeat(1,top_k), (h_t_rel_eval[target].shape[0], top_k, num_models))
        eval_features.append(target_eval)
    # selected_feature = torch.vstack(selected_feature)
    scores_repeat = torch.cat(selected_scores, 1)
    eval_repeat = torch.cat(eval_features, 1)
    signal_repeat = torch.cat([torch.zeros(eval_repeat.shape[0], top_k, 1), torch.ones(eval_repeat.shape[0], top_k, 1)], 1)
    selected_feature = torch.cat([signal_repeat, eval_repeat, scores_repeat], 2)
    selected_feature = selected_feature.reshape(selected_feature.shape[0] * selected_feature.shape[1], selected_feature.shape[2])
    selected_feature = torch.nan_to_num(selected_feature, -999.)
    return selected_feature


def generate_pred_input_feature(mapped_triples: MappedTriples, context_resource):
    head_scores = []
    tail_scores = []
    head_eval = []
    tail_eval = []
    num_models = len(context_resource['models'])
    for m in context_resource['models']:
        m_context = context_resource[m]
        m_preds = m_context['preds']
        m_h_preds, m_t_preds = torch.chunk(m_preds, 2, 1)
        head_scores.append(torch.flatten(m_h_preds, start_dim=0, end_dim=1))
        tail_scores.append(torch.flatten(m_t_preds, start_dim=0, end_dim=1))
        m_rel_h_t_eval = get_rel_eval_scores(mapped_triples, m_context['rel_eval'], context_resource['releval2idx'])
        m_h_eval, m_t_eval = torch.chunk(m_rel_h_t_eval, 2, 1)
        head_eval.append(m_h_eval)
        tail_eval.append(m_t_eval)

    candidate_num = int(head_scores[0].shape[0] / mapped_triples.shape[0])
    h_signal = torch.zeros(mapped_triples.shape[0], 1).repeat(candidate_num, 1)
    t_signal = torch.ones(mapped_triples.shape[0], 1).repeat(candidate_num, 1)
    head_eval = torch.hstack(head_eval).repeat((1, candidate_num)).reshape([candidate_num * mapped_triples.shape[0], num_models])
    tail_eval = torch.hstack(tail_eval).repeat((1, candidate_num)).reshape([candidate_num * mapped_triples.shape[0], num_models])
    head_scores = torch.vstack(head_scores).T
    tail_scores = torch.vstack(tail_scores).T
    h_features = torch.concat([h_signal, head_eval, head_scores], 1)
    t_features = torch.concat([t_signal, tail_eval, tail_scores], 1)
    h_t_features = torch.concat([h_features, t_features], 0)
    return h_t_features
