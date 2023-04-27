from abc import abstractmethod, ABC
import pykeen
from pykeen.datasets import get_dataset
from pykeen.utils import prepare_filter_triples
from context_load_and_run import ContextLoader
from typing import Optional
from pykeen.constants import TARGET_TO_INDEX
from pykeen.evaluation.evaluator import filter_scores_
from pykeen.evaluation.evaluator import create_sparse_positive_filter_
from pykeen.typing import MappedTriples, Target, LABEL_TAIL, LABEL_HEAD, COLUMN_HEAD, COLUMN_TAIL
from torch import FloatTensor
import torch
import pandas as pd


def get_all_pos_triples(dataset: pykeen.datasets.Dataset):
    all_pos_triples = prepare_filter_triples(
        mapped_triples=dataset.testing.mapped_triples,
        additional_filter_triples=[
            dataset.training.mapped_triples,
            dataset.validation.mapped_triples,
        ],
    )
    return all_pos_triples


class AnalysisChart(ABC):
    def __init__(self, params):
        self.params = params
        self.dataset = get_dataset(
            dataset=params.dataset
        )
        self.context_loader = ContextLoader(self.params.work_dir, self.params.models)
        self.all_pos_triples = get_all_pos_triples(self.dataset)

    @abstractmethod
    def analyze_test(self):
        pass

    @abstractmethod
    def get_partition_test_eval_per_model(self, target2tri_idx):
        pass

    @abstractmethod
    def get_partition_valid_eval_per_model(self, target2tri_idx):
        pass

    @abstractmethod
    def partition_eval_and_save(self, key2tri_ids):
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


def group_rank_eval(mapped_triples, group_idx_dict, scores_in_pykeen_format, all_pos_triples):
    # a light weight group evaluator
    h_score, t_score = scores_in_pykeen_format.chunk(2, 1)
    ht_scores = [h_score, t_score]
    targets = [LABEL_HEAD, LABEL_TAIL]
    # mask pos scores
    for i, target in enumerate(targets):
        ht_scores[i] = mask_positives(mapped_triples, ht_scores[i], target, all_pos_triples, -999)
    # eval groups
    head_tail_eval = {}
    for key, g_index in group_idx_dict.items():
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


def eval_groups(mapped_triples, ht_scores_in_pykeen_list, all_pos_triples):
    h_score, t_score = ht_scores_in_pykeen_list.chunk(2, 1)
    ht_scores = [h_score, t_score]
    targets = [LABEL_HEAD, LABEL_TAIL]
    # mask pos scores
    for i, target in enumerate(targets):
        ht_scores[i] = mask_positives(mapped_triples, ht_scores[i], target, all_pos_triples, -999)
    h_preds, t_preds = ht_scores[0], ht_scores[1]
    heads = mapped_triples[:, 0]
    tails = mapped_triples[:, 2]
    head_hits = calc_hit_at_k(h_preds, heads)
    tail_hits = calc_hit_at_k(t_preds, tails)
    both_hit = calc_hit_at_k(torch.cat([h_preds, t_preds]), torch.cat((heads, tails)))
    print(f"h[{head_hits}, t[{tail_hits}], b[{both_hit}]")
    return torch.as_tensor([head_hits, tail_hits, both_hit])


def find_relation_mappings(dataset: pykeen.datasets.Dataset):
    all_triples = torch.cat([dataset.training.mapped_triples,
                             dataset.validation.mapped_triples,
                             dataset.testing.mapped_triples], 0)
    df = pd.DataFrame(data=all_triples.numpy(), columns=['h', 'r', 't'])
    del all_triples
    hr2t = df.groupby(['h', 'r'], group_keys=True, as_index=False)['t'].nunique() # 't' column now is the number of unique ids
    hr2t_1 = hr2t.groupby('r', group_keys=True, as_index=False)['t'].sum() # r: number of tail entities
    hr2t_2 = hr2t.groupby('r', group_keys=True, as_index=False)['h'].nunique() # r: number of unique head entities

    rt2h = df.groupby(['r', 't'], group_keys=True, as_index=False)['h'].nunique()
    rt2h_1 = rt2h.groupby('r', group_keys=True, as_index=False)['h'].sum()
    rt2h_2 = rt2h.groupby('r', group_keys=True, as_index=False)['t'].nunique()

    ht_mappings = hr2t_2[['r']]
    ht_mappings['t2h'] = hr2t_1['t'] / hr2t_2['h']
    ht_mappings['h2t'] = rt2h_1['h'] / rt2h_2['t']

    one2one = ht_mappings[(ht_mappings['h2t'] < 1.5) & (ht_mappings['t2h'] < 1.5)]['r'].values
    one2n = ht_mappings[(ht_mappings['h2t'] < 1.5) & (ht_mappings['t2h'] >= 1.5)]['r'].values
    n2one = ht_mappings[(ht_mappings['h2t'] >= 1.5) & (ht_mappings['t2h'] < 1.5)]['r'].values
    n2m = ht_mappings[(ht_mappings['h2t'] >= 1.5) & (ht_mappings['t2h'] >= 1.5)]['r'].values
    rel_groups = {'1-1': one2one,
                  '1-n': one2n,
                  'n-1': n2one,
                  'n-m': n2m}
    return rel_groups


def calc_hit_at_k(pred_scores, ground_truth_idx):
    """Calculates mean number of hits@k. Higher values are ranked first.
    the pos scores have been masked
    Returns: list of float, of the same length as hit_positions, containing
        Hits@K score.

    """
    # scores[torch.arange(0, batch.shape[0]), batch[:, column]]
    targets = pred_scores[torch.arange(0, pred_scores.shape[0]), ground_truth_idx].unsqueeze(1)
    ranks = torch.zeros(pred_scores.shape[0])
    ranks[:] += torch.sum((pred_scores >= targets).float(), dim=1).cpu()
    hits_at = list(map(
        lambda x: torch.mean((ranks <= x).float()).item(),
        [1,3,10]
    ))
    mrr = torch.mean(1. / ranks).item()
    hits_at.append(mrr)
    return hits_at


def to_fusion_eval_format(dataset: pykeen.datasets.Dataset, eval_preds_in_pykeen, all_pos_triples, out_dir, top_k=10):
    torch.save(eval_preds_in_pykeen, out_dir + "valid_preds.pt")
    mapped_triples = dataset.validation.mapped_triples
    total_scores = eval_groups(mapped_triples, eval_preds_in_pykeen, all_pos_triples)
    torch.save(torch.as_tensor(total_scores), out_dir + "rank_total_eval.pt")
    m_dev_preds = torch.chunk(eval_preds_in_pykeen, 2, 1)
    pos_scores = m_dev_preds[0]
    pos_scores = pos_scores[torch.arange(0, dataset.validation.mapped_triples.shape[0]),
                            dataset.validation.mapped_triples[:, 0]]
    neg_scores, neg_index_topk = get_neg_scores_top_k(dataset.validation.mapped_triples, m_dev_preds, all_pos_triples, top_k) # [[h1 * candidate, h2 * candicate...][t1,t2...]]

    torch.save(pos_scores, out_dir + "eval_pos_scores.pt")
    torch.save(neg_scores, out_dir + "eval_neg_scores.pt")
    torch.save(neg_index_topk, out_dir + "eval_neg_index.pt")


def get_neg_scores_top_k(mapped_triples, dev_predictions, all_pos_triples, top_k):
    # Create filter
    targets = [COLUMN_HEAD, COLUMN_TAIL]
    neg_scores = []
    neg_index = []
    for index in range(2):
        # exclude positive triples
        positive_filter, _ = create_sparse_positive_filter_(
            hrt_batch=mapped_triples,
            all_pos_triples=all_pos_triples,
            relation_filter=None,
            filter_col=targets[index],
        )
        scores = filter_scores_(scores=dev_predictions[index], filter_batch=positive_filter)
        # pick top_k negs, if no more than top_k, then fill with -999.
        select_range = top_k if top_k < scores.shape[-1] else scores.shape[-1]
        scores_k, indices_k = torch.nan_to_num(scores, nan=-999.).topk(k=select_range)
        # remove -999 from scores_k
        nan_index = torch.nonzero(scores_k == -999.)
        indices_k[nan_index[:, 0], nan_index[:, 1]] = int(-1)
        neg_scores.append(scores_k)
        neg_index.append(indices_k)
    neg_scores = torch.stack(neg_scores, 1)  # [h1* candi,h2 * candi...,t1 * candi, t2* candi...]
    neg_index = torch.stack(neg_index, 1)
    return neg_scores, neg_index




