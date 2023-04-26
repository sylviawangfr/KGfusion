import argparse
import logging
import torch
from blenders.blender_utils import Blender
from context_load_and_run import load_score_context
from features.feature_scores_only_dataset import ScoresOnlyDataset
from lp_kge.lp_pykeen import get_all_pos_triples


logger = logging.getLogger(__name__)


class TNormBlender(Blender):
    def __init__(self, params, logger):
        super().__init__(params, logger)
        self.context = load_score_context(self.params.models,
                                          in_dir=params.work_dir,
                                          calibration=True
                                          )

    def aggregate_scores(self):
        work_dir = self.params.work_dir
        mapped_triples = self.dataset.testing.mapped_triples
        all_pos = get_all_pos_triples(self.dataset)
        test_data_feature = ScoresOnlyDataset(mapped_triples, self.context, all_pos)
        score_feature = test_data_feature.get_all_test_examples()
        scores_blender = torch.min(score_feature, dim=1).values
        option_str = f"{self.params.dataset}_{'_'.join(self.params.models)}_" \
                         "tnorm"
        self.finalize(scores_blender, option_str)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="ComplEx_RotatE_TuckER_CPComplEx_anyburl")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    args = parser.parse_args()
    args.models = args.models.split('_')
    wab = TNormBlender(args, logger)
    wab.aggregate_scores()
