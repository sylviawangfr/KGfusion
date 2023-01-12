from collections import defaultdict
from typing import List, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, TypeVar, Union, cast, Iterable
import numpy as np
import torch
from pykeen.evaluation.rank_based_evaluator import _iter_ranks, RankBasedMetricResults
from pykeen.evaluation.ranks import Ranks
from pykeen.metrics.ranking import rank_based_metric_resolver
from pykeen.evaluation import Evaluator
from pykeen.typing import Target, RankType, MappedTriples, ExtendedTarget, SIDE_BOTH, RANK_TYPES
import logging

from lp_kge.grouped_classification_evaluator import GroupedMetricResults

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

K = TypeVar("K")


def _flatten(nested: Mapping[K, Sequence[np.ndarray]]) -> Mapping[K, np.ndarray]:
    return {key: np.concatenate(value) for key, value in nested.items()}


def _select_index(nested: Mapping[K, np.ndarray], index: np.ndarray) -> Mapping[K, np.ndarray]:
    return {key: [[value[i] for i in list(index) if i < len(value)]] for key, value in nested.items()}


def _iter_ranks2(
        ranks: Mapping[Tuple[Target, RankType], Sequence[np.ndarray]],
        num_candidates: Mapping[Target, Sequence[np.ndarray]],
        weights: Optional[Mapping[Target, Sequence[np.ndarray]]] = None,
) -> Iterable[Tuple[ExtendedTarget, RankType, np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    # terminate early if there are no ranks
    if not ranks:
        logger.debug("Empty ranks. This should only happen during size probing.")
        return

    sides = sorted(num_candidates.keys())
    # flatten dictionaries
    ranks_flat = ranks
    num_candidates_flat = num_candidates
    weights_flat: Mapping[Target, np.ndarray]
    if weights is None:
        weights_flat = dict()
    else:
        weights_flat = _flatten(weights)
    for rank_type in RANK_TYPES:
        # individual side
        for side in sides:
            yield side, rank_type, ranks_flat[side, rank_type], num_candidates_flat[side], weights_flat.get(side)

        # combined
        c_ranks = np.concatenate([ranks_flat[side, rank_type] for side in sides])
        c_num_candidates = np.concatenate([num_candidates_flat[side] for side in sides])
        c_weights = None if weights is None else np.concatenate([weights_flat[side] for side in sides])
        yield SIDE_BOTH, rank_type, c_ranks, c_num_candidates, c_weights


class GroupedRankBasedEvaluator(Evaluator):
    """A rank-based evaluator for KGE models."""

    num_entities: Optional[int]
    ranks: MutableMapping[Tuple[Target, RankType], List[np.ndarray]]
    num_candidates: MutableMapping[Target, List[np.ndarray]]
    all_ranks: MutableMapping[Tuple[Target, RankType], np.ndarray]
    all_num_candidates: MutableMapping[Target, np.ndarray]

    def __init__(
            self,
            filtered: bool = True,
            **kwargs,
    ):
        """Initialize rank-based evaluator.

        :param filtered:
            Whether to use the filtered evaluation protocol. If enabled, ranking another true triple higher than the
            currently considered one will not decrease the score.
        :param metrics:
            the rank-based metrics to compute
        :param metrics_kwargs:
            additional keyword parameter
        :param add_defaults:
            whether to add all default metrics besides the ones specified by `metrics` / `metrics_kwargs`.
        :param kwargs: Additional keyword arguments that are passed to the base class.
        """
        super().__init__(
            filtered=filtered,
            requires_positive_mask=False,
            batch_size=32,
            automatic_memory_optimization=False,
            **kwargs,
        )
        self.metrics = rank_based_metric_resolver.make_many(['hitsatk'], [{'k': 10}])
        self.ranks = defaultdict(list)
        self.num_candidates = defaultdict(list)
        self.num_entities = None
        self.all_ranks = None
        self.all_num_candidates = None
        self.targets = []
        self.index_group = {}
        self.eval_triples = []

    # docstr-coverage: inherited
    def process_scores_(
            self,
            hrt_batch: MappedTriples,
            target: Target,
            scores: torch.FloatTensor,
            true_scores: Optional[torch.FloatTensor] = None,
            dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:  # noqa: D102
        if true_scores is None:
            raise ValueError(f"{self.__class__.__name__} needs the true scores!")

        batch_ranks = Ranks.from_scores(
            true_score=true_scores,
            all_scores=scores,
        )
        self.num_entities = scores.shape[1]
        for rank_type, v in batch_ranks.items():
            self.ranks[target, rank_type].append(v.detach().cpu().numpy())
        self.num_candidates[target].append(batch_ranks.number_of_options.detach().cpu().numpy())

    # docstr-coverage: inherited
    def finalize(self) -> GroupedMetricResults:  # noqa: D102
        if self.num_entities is None:
            raise ValueError
        self.all_ranks = _flatten(self.ranks)
        self.all_num_candidates = _flatten(self.num_candidates)
        group_results = []
        for g_index in self.index_group:
            g_rank = _select_index(self.all_ranks, g_index)
            g_num_candidates = _select_index(self.all_num_candidates, g_index)
            result = RankBasedMetricResults.from_ranks(
                metrics=self.metrics,
                rank_and_candidates=_iter_ranks(ranks=g_rank, num_candidates=g_num_candidates),
            )
            tmp_eval = result.data
            if len(self.targets) == 2:
                group_results.append([tmp_eval[('hits_at_10', 'head', 'realistic')],
                                      tmp_eval[('hits_at_10', 'tail', 'realistic')],
                                      tmp_eval[('hits_at_10', 'both', 'realistic')]])
            else:
                group_results.append([tmp_eval[('hits_at_10', self.targets[0], 'realistic')]])

        # Clear buffers
        self.ranks.clear()
        self.num_candidates.clear()
        self.all_ranks.clear()
        self.all_num_candidates.clear()
        return GroupedMetricResults({'rank': torch.as_tensor(group_results)})

    def set_groups(self, index_group, mapped_triples, targets):
        self.index_group = index_group
        self.eval_triples = mapped_triples
        self.targets = targets
        return self
