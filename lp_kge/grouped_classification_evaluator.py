# -*- coding: utf-8 -*-

"""Implementation of wrapper around sklearn metrics."""

from typing import Mapping, MutableMapping, Optional, Tuple, Type, cast

from pykeen.constants import TARGET_TO_INDEX
from pykeen.evaluation import Evaluator, ClassificationMetricResults, MetricResults
import numpy as np
import torch


__all__ = [
    "GroupededClassificationEvaluator",
]

from pykeen.typing import Target, MappedTriples


class GroupedMetricResults(MetricResults):
    def get_metric(self, name: str) -> float:  # noqa: D102
        return self.data[name]


class GroupededClassificationEvaluator(Evaluator):
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
            batch_size=32,
            automatic_memory_optimization=False,
            **kwargs,
        )
        self.targets = []
        self.all_scores = {}
        self.all_positives = {}
        self.index_group = {}
        self.eval_triples = []
        self.counts = 0

    # docstr-coverage: inherited
    def process_scores_(
        self,
        hrt_batch: MappedTriples,
        target: Target,
        scores: torch.FloatTensor,
        true_scores: Optional[torch.FloatTensor] = None,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:
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
            self.counts += 1

    def _get_group_scores_and_positives(self, g_triples, tmp_targets):
        g_scores = {}
        g_positives = {}
        for target in tmp_targets:
            remaining = [i for i in range(g_triples.shape[1]) if i != TARGET_TO_INDEX[target]]
            keys = g_triples[:, remaining]
            # Ensure that each key gets counted only once
            for i in range(keys.shape[0]):
                # include head_side flag into key to differentiate between (h, r) and (r, t)
                key_suffix = tuple(map(int, keys[i]))
                assert len(key_suffix) == 2
                key_suffix = cast(Tuple[int, int], key_suffix)
                key = (target,) + key_suffix
                if key not in self.all_scores.keys():
                    continue  # pykeen evaluator only do same size batches, and ignore the last a few triples.
                g_scores[key] = self.all_scores[key]
                g_positives[key] = self.all_positives[key]
        tmp_keys = list(g_scores.keys())
        g_y_score = np.concatenate([g_scores[k] for k in tmp_keys], axis=0).flatten()
        g_y_true = np.concatenate([g_positives[k] for k in tmp_keys], axis=0).flatten()
        tmp_result = ClassificationMetricResults.from_scores(g_y_true, g_y_score)
        return tmp_result.data['f1_score']

    # docstr-coverage: inherited
    def finalize(self) -> GroupedMetricResults:  # noqa: D102
        # Because the order of the values of an dictionary is not guaranteed,
        # we need to retrieve scores and masks using the exact same key order.
        print("keys/counts:")
        print(len(self.all_scores.keys()))
        print(str(self.counts))
        all_f1 = []
        all_keys = list(self.all_scores.keys())
        if len(all_keys) > 0:
            # y_score = np.concatenate([self.all_scores[k] for k in all_keys], axis=0).flatten()
            # y_true = np.concatenate([self.all_positives[k] for k in all_keys], axis=0).flatten()
            for g_index in self.index_group:
                g_triples = self.eval_triples[g_index, :]
                if len(self.targets) == 2:
                    g_f1 = []
                    for t in self.targets:
                        g_f1.append(self._get_group_scores_and_positives(g_triples, [t]))
                    g_f1.append(self._get_group_scores_and_positives(g_triples, self.targets))
                    all_f1.append(g_f1)
                else:
                    all_f1.append(self._get_group_scores_and_positives(g_triples, self.targets))
            ts_all_f1 = torch.as_tensor(all_f1)
            result = GroupedMetricResults({'f1': ts_all_f1})
            # Clear buffers
            self.all_positives.clear()
            self.all_scores.clear()
            self.index_group.clear()
            self.targets.clear()
        else:
            result = GroupedMetricResults({})
        return result

    def set_groups(self, index_group, mapped_triples, targets):
        self.index_group = index_group
        self.eval_triples = mapped_triples
        self.targets = targets
        return self



