from abc import abstractmethod, ABC

from pykeen.datasets import get_dataset

from context_load_and_run import load_score_context
from lp_kge.lp_pykeen import get_all_pos_triples
from typing import Optional
from pykeen.constants import TARGET_TO_INDEX
from pykeen.evaluation.evaluator import filter_scores_
from pykeen.evaluation.evaluator import create_sparse_positive_filter_
from pykeen.typing import MappedTriples, Target, LABEL_TAIL, LABEL_HEAD
from torch import FloatTensor
import torch
import pandas as pd


class AnalysisChart(ABC):
    def __init__(self, params):
        self.params = params
        self.dataset = get_dataset(
            dataset=params.dataset
        )
        self.context = load_score_context(self.params.models,
                                          in_dir=self.params.work_dir,
                                          calibration=self.params.cali == 'True'
                                          )
        self.all_pos_triples = get_all_pos_triples(self.dataset)

    @abstractmethod
    def analyze(self):
        pass


def mask_positives(batch: MappedTriples,
                   target_scores: FloatTensor,
                   target: Target,
                   all_pos_triples: Optional[MappedTriples], mask_value=torch.nan):
    column = TARGET_TO_INDEX[target]
    positive_filter, relation_filter = create_sparse_positive_filter_(
        hrt_batch=batch,
        all_pos_triples=all_pos_triples,
        relation_filter=None,
        filter_col=column,
    )
    true_scores = target_scores[torch.arange(0, batch.shape[0]), batch[:, column]]
    # overwrite filtered scores
    scores = filter_scores_(scores=target_scores, filter_batch=positive_filter)
    # The scores for the true triples have to be rewritten to the scores tensor
    scores[torch.arange(0, batch.shape[0]), batch[:, column]] = true_scores
    # the rank-based evaluators needs the true scores with trailing 1-dim
    torch.nan_to_num(scores, nan=mask_value)
    return scores


def group_rank_eval(dataset, mapped_triples, group_idx_iter, scores_in_pykeen_format):
    # a light weight group evaluator
    all_pos = get_all_pos_triples(dataset)
    h_score, t_score = scores_in_pykeen_format.chunk(2, 1)
    ht_scores = [h_score, t_score]
    targets = [LABEL_HEAD, LABEL_TAIL]
    # mask pos scores
    for i, target in enumerate(targets):
        ht_scores[i] = mask_positives(mapped_triples, ht_scores[i], target, all_pos, -999)
    # eval groups
    head_tail_eval = {}
    for key, g_index in group_idx_iter.items():
        if len(g_index) == 0:
            head_tail_eval.update({key: torch.zeros((4, 3))})
        else:
            g_triples = mapped_triples[g_index]
            g_heads = g_triples[:, 0]
            g_tails = g_triples[:, 2]
            g_h_preds = ht_scores[0][g_index]
            g_t_preds = ht_scores[1][g_index]
            head_hits = calc_hit_at_k(g_h_preds, g_heads)
            tail_hits = calc_hit_at_k(g_t_preds, g_tails)
            both_hit = calc_hit_at_k(torch.cat([g_h_preds, g_t_preds]), torch.cat((g_heads, g_tails)))
            head_tail_eval.update({key: torch.as_tensor([head_hits,
                                                         tail_hits,
                                                         both_hit])})
    return head_tail_eval


def calc_hit_at_k(masked_pred_scores, ground_truth_idx):
    """Calculates mean number of hits@k. Higher values are ranked first.
    the pos scores have been masked
    Returns: list of float, of the same length as hit_positions, containing
        Hits@K score.

    """
    targets = masked_pred_scores[torch.arange(0, masked_pred_scores.shape[0]), ground_truth_idx].unsqueeze(1)
    ranks = torch.zeros(masked_pred_scores.shape[0])
    ranks[:] += torch.sum((masked_pred_scores >= targets).float(), dim=1).cpu()
    hits_at = list(map(
        lambda x: torch.mean((ranks <= x).float()).item(),
        [1, 3, 10]
    ))
    mrr = torch.mean(1. / ranks).item()
    hits_at.append(mrr)
    return hits_at


