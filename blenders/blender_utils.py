from typing import Optional
import pandas as pd
import pykeen
from pykeen.constants import TARGET_TO_INDEX
from pykeen.evaluation import RankBasedEvaluator
from pykeen.evaluation.evaluator import Evaluator, filter_scores_
from pykeen.evaluation.evaluator import create_sparse_positive_filter_, create_dense_positive_mask_
from pykeen.typing import MappedTriples, Target, LABEL_HEAD, LABEL_TAIL, COLUMN_HEAD, COLUMN_TAIL
from pykeen.utils import prepare_filter_triples
from torch import FloatTensor
import torch
import common_utils
from pykeen.datasets import get_dataset
from analysis.group_eval_utils import eval_groups, get_all_pos_triples


def evaluate_target(
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
    if evaluator.filtered:
        assert positive_filter is not None
        # Select scores of true
        true_scores = scores[torch.arange(0, batch.shape[0]), batch[:, column]]
        # overwrite filtered scores
        scores = filter_scores_(scores=scores, filter_batch=positive_filter)
        # The scores for the true triples have to be rewritten to the scores tensor
        scores[torch.arange(0, batch.shape[0]), batch[:, column]] = true_scores
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


def evaluate_testing_scores(dataset, pred_scores_in_pykeen_format):
    all_pos = get_all_pos_triples(dataset)
    ht_scores = torch.chunk(pred_scores_in_pykeen_format, 2, 1)
    evaluator = RankBasedEvaluator()
    relation_filter = None
    for ind, target in enumerate([LABEL_HEAD, LABEL_TAIL]):
        relation_filter = evaluate_target(
            batch=dataset.testing.mapped_triples,
            scores=ht_scores[ind],
            target=target,
            evaluator=evaluator,
            all_pos_triples=all_pos,
            relation_filter=relation_filter,
        )
    result = evaluator.finalize()
    str_re = common_utils.format_result(result)
    return str_re


# def per_rel_eval(mapped_triples, scores, out_dir):
#     triples_df = pd.DataFrame(data=mapped_triples.numpy(), columns=['h', 'r', 't'])
#     original_groups = triples_df.groupby('r', group_keys=True, as_index=False)
#     group_keys = original_groups.groups.keys()
#     file_name = "rank_rel_eval.pt"
#     head_tail_eval = []
#     for rel in group_keys:
#         # generate grouped index of eval mapped triples
#         g = original_groups.get_group(rel)
#         g_index = torch.from_numpy(g.index.values)
#         rg_index = torch.as_tensor(g_index)
#         h_preds, t_preds = scores[rg_index].chunk(2, 1)
#         heads = torch.as_tensor(g['h'].values)
#         tails = torch.as_tensor(g['t'].values)
#         head_hits = calc_hit_at_k(h_preds, heads)
#         tail_hits = calc_hit_at_k(t_preds, tails)
#         both_hit = calc_hit_at_k(torch.cat([h_preds, t_preds]), torch.cat((heads, tails)))
#         head_tail_eval.append(torch.as_tensor([head_hits,
#                                                tail_hits,
#                                                both_hit]))
#     head_tail_eval = torch.stack(head_tail_eval, 0)
#     torch.save(head_tail_eval, out_dir + file_name)


# def per_mapping_eval(pykeen_dataset, scores, out_dir):
#     mappings = ['1-1', '1-n', 'n-1', 'n-m']
#     rel_mappings = find_relation_mappings(pykeen_dataset)
#     dev = pykeen_dataset.validation.mapped_triples
#     triples_df = pd.DataFrame(data=dev.numpy(), columns=['h', 'r', 't'], )
#     relmapping2idx = dict()
#     for idx, mapping_type in enumerate(mappings):
#         mapped_rels = rel_mappings[mapping_type]
#         relmapping2idx.update({int(rel): idx for rel in mapped_rels})
#     head_tail_eval = []
#     for rel_group in mappings:
#         tmp_rels = rel_mappings[rel_group]
#         tri_group = triples_df.query('r in @tmp_rels')
#         if len(tri_group.index) > 0:
#             g = torch.from_numpy(tri_group.values)
#             g_index = torch.from_numpy(tri_group.index.values)
#             rg_index = torch.as_tensor(g_index)
#             h_preds, t_preds = scores[rg_index].chunk(2, 1)
#             heads = g[:, 0]
#             tails = g[:, 2]
#             head_hits = calc_hit_at_k(h_preds, heads)
#             tail_hits = calc_hit_at_k(t_preds, tails)
#             both_hit = calc_hit_at_k(torch.cat([h_preds, t_preds]), torch.cat((heads, tails)))
#             head_tail_eval.append(torch.as_tensor([head_hits,
#                                                    tail_hits,
#                                                    both_hit]))
#         else:
#             head_tail_eval.append(torch.full([3, 4], 0.001))
#     torch.save(torch.stack(head_tail_eval, 0), out_dir + f"rank_mapping_rel_eval.pt")
    # if not common_utils.does_exist(out_dir + f"rank_mapping_releval2idx.json"):
    #     common_utils.save2json(relmapping2idx, out_dir + f"rank_mapping_releval2idx.json")


#
# def per_mapping_eval(dataset, pred_scores, out_dir):
#     mappings = ['1-1', '1-n', 'n-1', 'n-m']
#     rel_mappings = find_relation_mappings(dataset)
#     tri_df = pd.DataFrame(data=dataset.validation.mapped_triples.numpy(), columns=['h', 'r', 't'])
#     head_tail_mrr = []
#     for rel_group in mappings:
#         tmp_rels = rel_mappings[rel_group]
#         rg_tris = tri_df.query('r in @tmp_rels')
#         if len(rg_tris.index) > 0:
#             rg_index = torch.as_tensor(rg_tris.index)
#             h_preds, t_preds = pred_scores[rg_index].chunk(2, 1)
#             heads = torch.as_tensor(rg_tris['h'].values)
#             tails = torch.as_tensor(rg_tris['t'].values)
#             head_hits = calc_hit_at_k(h_preds, heads)
#             tail_hits = calc_hit_at_k(t_preds, tails)
#             both_hit = calc_hit_at_k(torch.cat([h_preds, t_preds]), torch.cat((heads, tails)))
#             head_tail_mrr.append(torch.as_tensor([head_hits,
#                                                   tail_hits,
#                                                   both_hit]))
#         else:
#             head_tail_mrr.append(torch.full([3, 4], 0.001))
#     head_tail_mrr = torch.stack(head_tail_mrr, 0)
#     torch.save(head_tail_mrr, out_dir + "rank_mapping_rel_eval.pt")





