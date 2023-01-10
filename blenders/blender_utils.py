from typing import Optional
from pykeen.constants import TARGET_TO_INDEX
from pykeen.evaluation.evaluator import Evaluator, filter_scores_
from pykeen.evaluation.evaluator import create_sparse_positive_filter_, create_dense_positive_mask_
from pykeen.typing import MappedTriples, Target
from torch import FloatTensor
import torch
from features.feature_per_ent_dataset import PerEntDataset
from features.feature_per_rel_both_dataset import PerRelBothDataset
from features.feature_per_rel_dataset import PerRelDataset
from features.feature_per_rel_ent_dataset import PerRelEntDataset
from features.feature_scores_only_dataset import ScoresOnlyDataset
from abc import ABC, abstractmethod
from pykeen.datasets import get_dataset


class Blender(ABC):
    def __init__(self, params):
        self.dataset = get_dataset(
            dataset=params['dataset']
        )
        self.params = params


    @abstractmethod
    def aggregate_scores(self):
        pass


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


def get_features_clz(keyword=2):
    clz = {
        1: PerRelDataset,
        2: PerRelBothDataset,
        3: ScoresOnlyDataset,
        4: PerEntDataset,
        5: PerRelEntDataset}
    return clz[keyword]

