import argparse
import logging
import torch
from blenders.blender_base import Blender, get_features_clz
from context_load_and_run import ContextLoader
from lp_kge.lp_pykeen import get_all_pos_triples

logger = logging.getLogger(__name__)


def weighted_mean(t1, weights):
    eval_mul_score = torch.sum(torch.mul(t1, weights), 1)
    weights_sum = torch.sum(weights, 1)
    weights_sum[weights_sum == 0] = 0.5  # do not divided by zero
    tmp_blender = torch.div(eval_mul_score, weights_sum)
    return tmp_blender


class PickTheBestBlender(Blender):
    def __init__(self, params, logger):
        super().__init__(params, logger)
        self.context_loader = ContextLoader(in_dir=params.work_dir, model_list=params.models)

    def aggregate_scores(self):
        all_pos_triples = get_all_pos_triples(self.dataset)
        get_features = get_features_clz(self.params.features)
        test_data_feature = get_features(self.dataset.testing.mapped_triples,
                                         self.context_loader,
                                         all_pos_triples,
                                         calibrated=True)
        eval_feature, score_feature = torch.chunk(test_data_feature.get_all_test_examples(), 2, 1)
        # assign lower eval scores to zero, so that we pick the best performed model
        max_values = torch.max(eval_feature, dim=1).values.unsqueeze(1)
        max_index = torch.nonzero(eval_feature != max_values, as_tuple=True)
        eval_feature[max_index] = 0

        blender_scores = weighted_mean(score_feature, eval_feature)

        option_str = f"{self.params.dataset}_{'_'.join(self.params.models)}_" \
                     f"data{self.params.features}" \
                     f"_{self.params.eval_feature}" \
                     f"_best"
        self.finalize(blender_scores, option_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="ComplEx_CP_RotatE_TuckER_anyburl")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    parser.add_argument('--eval_feature', type=str, default='rel')
    # "1": PerRelDataset,
    # "2": PerRelBothDataset,
    # "4": PerEntDegreeDataset,
    # "6": PerModelBothDataset
    parser.add_argument('--features', type=int, default=1)  # 1, 2, 4, 6
    args = parser.parse_args()
    args.models = args.models.split('_')
    wab = PickTheBestBlender(args, logging.getLogger('PickTheBestBlender'))
    wab.aggregate_scores()
