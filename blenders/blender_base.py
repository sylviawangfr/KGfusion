import logging
from abc import ABC, abstractmethod

import torch
from pykeen.datasets import get_dataset

import common_utils
from blenders.blender_utils import evaluate_testing_scores
from features.feature_per_ent_dataset import PerEntDegreeDataset
from features.feature_per_model_both_dataset import PerModelBothDataset
from features.feature_per_rel_both_dataset import PerRelBothDataset
from features.feature_per_rel_ht_dataset import PerRelDataset
from features.feature_scores_only_dataset import ScoresOnlyDataset


class Blender(ABC):
    def __init__(self, params, logger):
        self.dataset = get_dataset(
            dataset=params.dataset
        )
        self.params = params
        self.log_dir = self.params.work_dir + "logs/"
        self.logger = logger
        common_utils.init_dir(self.log_dir)
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)


    @abstractmethod
    def aggregate_scores(self):
        pass

    def finalize(self, blended_preds, log_prefix):
        h_preds, t_preds = torch.chunk(blended_preds, 2, 0)
        # restore format that required by pykeen evaluator
        candidate_number = self.dataset.num_entities
        h_preds = torch.reshape(h_preds, (self.dataset.testing.num_triples, candidate_number))
        t_preds = torch.reshape(t_preds, (self.dataset.testing.num_triples, candidate_number))
        ht_scores = torch.cat([h_preds, t_preds], 1)
        save_dir = self.params.work_dir + '_'.join(self.params.models) + '/'
        common_utils.init_dir(save_dir)
        file_name = save_dir + f"{log_prefix}.pt"
        torch.save(ht_scores, file_name)
        self.logger.info(f"Transforming saved at {file_name}")
        str_re = evaluate_testing_scores(self.dataset, ht_scores)
        common_utils.save_to_file(str_re, self.log_dir + log_prefix + '.log')
        self.logger.info(f"{log_prefix}:\n{str_re}")


def get_features_clz(keyword=2):
    clz = {
        1: PerRelDataset,
        2: PerRelBothDataset,
        3: ScoresOnlyDataset,
        4: PerEntDegreeDataset,
        6: PerModelBothDataset}
    return clz[keyword]