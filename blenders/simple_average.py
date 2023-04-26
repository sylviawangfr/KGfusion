import argparse
import logging

import torch
from pykeen.evaluation import RankBasedEvaluator
from pykeen.typing import LABEL_HEAD, LABEL_TAIL
from pykeen.utils import resolve_device

import common_utils
from blenders.blender_utils import evaluate_target, evaluate_testing_scores
from blenders.blender_base import Blender
from common_utils import format_result, save_to_file
from context_load_and_run import load_score_context
from features.feature_scores_only_dataset import ScoresOnlyDataset
from lp_kge.lp_pykeen import get_all_pos_triples, predict_head_tail_scores


class SimpleAverageBlender(Blender):
    def __init__(self, params, logger):
        super().__init__(params, logger)
        self.context = load_score_context(self.params.models,
                                          in_dir=params.work_dir,
                                          calibration=params.cali=='True'
                                          )

    def aggregate_scores(self):
        mapped_triples = self.dataset.testing.mapped_triples
        all_pos = get_all_pos_triples(self.dataset)
        test_data_feature = ScoresOnlyDataset(mapped_triples, self.context, all_pos)
        score_feature = test_data_feature.get_all_test_examples()
        blender_scores = torch.mean(score_feature, dim=1)
        option_str = f"{self.params.dataset}_{'_'.join(self.params.models)}_" \
                     "savg"
        self.finalize(blender_scores, option_str)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="anyburl_CPComplEx")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    parser.add_argument('--cali', type=str, default="True")
    args = parser.parse_args()
    args.models = args.models.split('_')
    wab = SimpleAverageBlender(args, logging.getLogger(__name__))
    wab.aggregate_scores()
