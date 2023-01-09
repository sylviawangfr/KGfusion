# -*- coding: utf-8 -*-

"""Implementation of wrapper around sklearn metrics."""

from typing import Mapping, MutableMapping, Optional, Tuple, Type, cast

from pykeen.constants import TARGET_TO_INDEX
from pykeen.evaluation import Evaluator, ClassificationMetricResults
import numpy as np
import torch


__all__ = [
    "PatchedClassificationEvaluator",
]

from pykeen.typing import Target, MappedTriples


class PatchedClassificationEvaluator(Evaluator):
    """An evaluator that uses a classification metrics."""

    all_scores: MutableMapping[Tuple[Target, int, int], np.ndarray]
    all_positives: MutableMapping[Tuple[Target, int, int], np.ndarray]

    def __init__(self, **kwargs):
        """
        Initialize the evaluator.

        :param kwargs:
            keyword-based parameters passed to :meth:`Evaluator.__init__`.
        """
        super().__init__(
            filtered=False,
            requires_positive_mask=True,
            **kwargs,
        )
        self.all_scores = {}
        self.all_positives = {}

    # docstr-coverage: inherited
    def process_scores_(
        self,
        hrt_batch: MappedTriples,
        target: Target,
        scores: torch.FloatTensor,
        true_scores: Optional[torch.FloatTensor] = None,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:  # noqa: D102
        if dense_positive_mask is None:
            raise KeyError("Sklearn evaluators need the positive mask!")

        # Transfer to cpu and convert to numpy
        scores = scores.detach().cpu().numpy()
        dense_positive_mask = dense_positive_mask.detach().cpu().numpy()
        remaining = [i for i in range(hrt_batch.shape[1]) if i != TARGET_TO_INDEX[target]]
        keys = hrt_batch[:, remaining].detach().cpu().numpy()

        # Ensure that each key gets counted only once
        for i in range(keys.shape[0]):
            # include head_side flag into key to differentiate between (h, r) and (r, t)
            key_suffix = tuple(map(int, keys[i]))
            assert len(key_suffix) == 2
            key_suffix = cast(Tuple[int, int], key_suffix)
            key = (target,) + key_suffix
            self.all_scores[key] = scores[i]
            self.all_positives[key] = dense_positive_mask[i]

    # docstr-coverage: inherited
    def finalize(self) -> ClassificationMetricResults:  # noqa: D102
        # Because the order of the values of an dictionary is not guaranteed,
        # we need to retrieve scores and masks using the exact same key order.
        all_keys = list(self.all_scores.keys())
        if len(all_keys) > 0:
            y_score = np.concatenate([self.all_scores[k] for k in all_keys], axis=0).flatten()
            y_true = np.concatenate([self.all_positives[k] for k in all_keys], axis=0).flatten()
            result = ClassificationMetricResults.from_scores(y_true, y_score)
        else:
            result = ClassificationMetricResults({})
        # Clear buffers
        self.all_positives.clear()
        self.all_scores.clear()
        return result
