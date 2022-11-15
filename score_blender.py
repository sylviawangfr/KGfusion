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

from tqdm import tqdm, trange

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def train_linear_blender(in_dim, pos_eval_and_scores, neg_eval_and_scores, params, work_dir):
    model = nn.Linear(in_features=in_dim, out_features=1)
    criterion = nn.MarginRankingLoss(10, reduction='sum')
    # criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    num_epochs = params['epochs']
    training_iter = trange(1, num_epochs+1)
    for e in training_iter:
        pos_out = model(pos_eval_and_scores)
        neg_out = model(neg_eval_and_scores)
        loss = criterion(pos_out.squeeze(), neg_out.view(len(pos_out), -1, ).mean(dim=1), torch.Tensor([1]))
        print('Loss at epoch {} : {}'.format(e, loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_iter.set_postfix(
            {
                "loss": loss
            }
        )
    torch.save(model, os.path.join(work_dir,
                                   f'{"_".join(params["models"])}_ensemble.pth'))
    return model


def get_neg_scores_for_training(
        triples: MappedTriples,
        predictions: [],  # [head_pred, tail_pred]
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
    head_scores_neg = []
    tail_scores_neg = []
    head_scores_pos = []
    tail_scores_pos = []
    head_eval = []
    tail_eval = []
    head_scores = []
    tail_scores = []
    head_neg_index_topk = []
    tail_neg_index_topk = []
    num_models = len(context_resource['models'])
    for m in context_resource['models']:
        m_context = context_resource[m]
        m_preds = m_context['eval']
        m_h_preds, m_t_preds = torch.chunk(m_preds, 2, 1)
        m_h_pos_scores = m_h_preds[torch.arange(0, mapped_triples.shape[0]), mapped_triples[:, COLUMN_HEAD]]
        m_t_pos_scores = m_t_preds[torch.arange(0, mapped_triples.shape[0]), mapped_triples[:, COLUMN_TAIL]]
        head_scores_pos.append(m_h_pos_scores)
        tail_scores_pos.append(m_t_pos_scores)
        # list of tensors, corresponding to the triples
        m_neg_scores, m_neg_index_topk2 = get_neg_scores_for_training(mapped_triples, [m_h_preds, m_t_preds], all_pos_triples, num_neg)
        head_scores_neg.append(torch.flatten(m_neg_scores[0], start_dim=0, end_dim=1))
        tail_scores_neg.append(torch.flatten(m_neg_scores[1], start_dim=0, end_dim=1))
        head_neg_index_topk.append(m_neg_index_topk2[0])
        tail_neg_index_topk.append(m_neg_index_topk2[1])
        # list of h,t eval scores, corresponding to the relations, in order of triples
        # [h_score, r_score]
        head_scores.append(torch.flatten(m_h_preds, start_dim=0, end_dim=1))
        tail_scores.append(torch.flatten(m_t_preds, start_dim=0, end_dim=1))
        m_rel_h_t_eval = get_rel_eval_scores(mapped_triples, m_context['rel_eval'], context_resource['releval2idx'])
        m_h_eval, m_t_eval = torch.chunk(m_rel_h_t_eval, 2, 1)
        head_eval.append(m_h_eval)
        tail_eval.append(m_t_eval)

    # eval features
    head_scores_pos = torch.vstack(head_scores_pos).T
    tail_scores_pos = torch.vstack(tail_scores_pos).T
    head_eval = torch.hstack(head_eval)
    tail_eval = torch.hstack(tail_eval)
    # pos feature
    h_pos_features = torch.concat([torch.zeros(mapped_triples.shape[0], 1), head_eval, head_scores_pos], 1)
    t_pos_features = torch.concat([torch.ones(mapped_triples.shape[0], 1), tail_eval, tail_scores_pos], 1)
    pos_feature = torch.concat([h_pos_features, t_pos_features], 0)
    # neg feature
    head_scores_neg = torch.vstack(head_scores_neg).T
    tail_scores_neg = torch.vstack(tail_scores_neg).T
    candidate_num = int(head_scores_neg.shape[0] / mapped_triples.shape[0])
    neg_h_signal = torch.zeros(mapped_triples.shape[0], 1).repeat((candidate_num, 1))
    neg_t_signal = torch.ones(mapped_triples.shape[0], 1).repeat((candidate_num, 1))
    neg_h_eval = head_eval.repeat((1, candidate_num)).reshape([candidate_num * mapped_triples.shape[0], num_models])
    neg_t_eval = tail_eval.repeat((1, candidate_num)).reshape([candidate_num * mapped_triples.shape[0], num_models])
    h_neg_features = torch.concat([neg_h_signal, neg_h_eval, head_scores_neg], 1)
    t_neg_features = torch.concat([neg_t_signal, neg_t_eval, tail_scores_neg], 1)
    neg_feature = [h_neg_features.reshape([mapped_triples.shape[0], candidate_num, h_neg_features.shape[1]]),
                   t_neg_features.reshape([mapped_triples.shape[0], candidate_num, t_neg_features.shape[1]])]
    neg_feature = get_multi_model_neg_topk(neg_feature, [head_neg_index_topk, tail_neg_index_topk], num_neg)
    return pos_feature, neg_feature


def get_multi_model_neg_topk(neg_feature_all, neg_index_topk, top_k):
    selected_feature = []
    for target in range(2):
        neg_feature = neg_feature_all[target]
        topk_idx = neg_index_topk[target]
        # count top_k frequent index
        topk_idx = torch.hstack(topk_idx)
        idx_df = pd.DataFrame(data=topk_idx.T.numpy())
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
        neg_feature = neg_feature[torch.arange(0, neg_feature.shape[0]).unsqueeze(1), fill_topk]
        selected_feature.append(neg_feature)
    selected_feature = torch.vstack(selected_feature)
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


def aggregate_scores(model, mapped_triples: MappedTriples, context_resource, all_pos_triples):
    h_t_features = generate_pred_input_feature(mapped_triples, context_resource)
    # Send to device
    device: torch.device = resolve_device()
    logger.info(f"Using device: {device}")
    if device is not None:
        model = model.to(device)
    # Ensure evaluation mode
    model.eval()
    # Send tensors to device
    h_t_features = h_t_features.to(device=device)
    ens_scores = model(h_t_features)
    #  unwrap scores
    h_scores, t_scores = torch.chunk(ens_scores, 2, 0)
    # restore format that required by pykeen evaluator
    evaluator = RankBasedEvaluator()
    relation_filter = None
    relation_filter = restore_eval_format(
        batch=mapped_triples,
        scores=h_scores,
        target='head',
        evaluator=evaluator,
        all_pos_triples=all_pos_triples,
        relation_filter=relation_filter,
    )
    relation_filter = restore_eval_format(
        batch=mapped_triples,
        scores=t_scores,
        target='tail',
        evaluator=evaluator,
        all_pos_triples=all_pos_triples,
        relation_filter=relation_filter,
    )
    result = evaluator.finalize()
    return result


def restore_eval_format(
        batch: MappedTriples,
        scores: FloatTensor,
        target: Target,
        evaluator: Evaluator,
        all_pos_triples: Optional[MappedTriples],
        relation_filter: Optional[torch.BoolTensor],
) -> torch.BoolTensor:
    """
    Evaluate ranking for batch.
    :param scores:
    :param batch: shape: (batch_size, 3)
        The batch of currently evaluated triples.
    :param target:
        The prediction target.
    :param evaluator:
        The evaluator
    :param all_pos_triples:
        All positive triples (required if filtering is necessary).
    :param relation_filter:
        The relation filter. Can be re-used.
    :raises ValueError:
        if all positive triples are required (either due to filtered evaluation, or requiring dense masks).

    :return:
        The relation filter, which can be re-used for the same batch.
    """
    if evaluator.filtered or evaluator.requires_positive_mask:
        column = TARGET_TO_INDEX[target]
        if all_pos_triples is None:
            raise ValueError(
                "If filtering_necessary of positive_masks_required is True, all_pos_triples has to be "
                "provided, but is None."
            )

        # Create filter
        positive_filter, relation_filter = create_sparse_positive_filter_(
            hrt_batch=batch,
            all_pos_triples=all_pos_triples,
            relation_filter=relation_filter,
            filter_col=column,
        )
    else:
        positive_filter = relation_filter = None
    candidate_number = int(scores.shape[0] / batch.shape[0])
    scores = scores.reshape([batch.shape[0], candidate_number])
    if evaluator.filtered:
        assert positive_filter is not None
        # Select scores of true
        true_scores = scores[torch.arange(0, batch.shape[0]), batch[:, column]]
        # the rank-based evaluators needs the true scores with trailing 1-dim
        true_scores = true_scores.unsqueeze(dim=-1)
    else:
        true_scores = None

    # Create a positive mask with the size of the scores from the positive filter
    if evaluator.requires_positive_mask:
        assert positive_filter is not None
        positive_mask = create_dense_positive_mask_(zero_tensor=torch.zeros_like(scores), filter_batch=positive_filter)
    else:
        positive_mask = None

    # process scores
    evaluator.process_scores_(
        hrt_batch=batch,
        target=target,
        true_scores=true_scores,
        scores=scores,
        dense_positive_mask=positive_mask,
    )

    return relation_filter
