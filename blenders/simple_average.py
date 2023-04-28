import argparse
import logging

import torch
from blenders.blender_base import Blender
from context_load_and_run import ContextLoader
from features.feature_scores_only_dataset import ScoresOnlyDataset
from lp_kge.lp_pykeen import get_all_pos_triples


class SimpleAverageBlender(Blender):
    def __init__(self, params, logger):
        super().__init__(params, logger)
        self.context_loader = ContextLoader(in_dir=params.work_dir, model_list=params.models)

    def aggregate_scores(self):
        all_pos_triples = get_all_pos_triples(self.dataset)
        test_data_feature = ScoresOnlyDataset(self.dataset.testing.mapped_triples,
                                              self.context_loader,
                                              all_pos_triples,
                                              calibrated=self.params.calibrated == 'True')
        score_feature = test_data_feature.get_all_test_examples()
        blender_scores = torch.mean(score_feature, dim=1)
        option_str = f"{self.params.dataset}_{'_'.join(self.params.models)}_" \
                     "savg"
        self.finalize(blender_scores, option_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="ComplEx_CP_RotatE_TuckER_anyburl")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    parser.add_argument('--calibrated', type=str, default="True")
    args = parser.parse_args()
    args.models = args.models.split('_')
    wab = SimpleAverageBlender(args, logging.getLogger(__name__))
    wab.aggregate_scores()
