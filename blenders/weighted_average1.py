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


def weighted_harmonic_mean(t1, weights):
    # normalize weights
    tmp_evl_sum = torch.sum(weights, 1)
    t1[t1 == 0] = 0.001  # do not divide by zero
    tmp_reciprocal = torch.div(weights, t1)
    tmp_reciprocal_sum = torch.sum(tmp_reciprocal, 1)
    # calculate weighted harmonic mean
    n_avg = torch.div(tmp_evl_sum, tmp_reciprocal_sum)
    return n_avg


class WeightedAverageBlender1(Blender):
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
        blender_scores = weighted_mean(score_feature, eval_feature)

        option_str = f"{self.params.dataset}_{'_'.join(self.params.models)}_" \
                     f"data{self.params.features}" \
                     f"_wavg"
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
    # "5": PerRelEntDataset
    # "6": PerModelBothDataset
    parser.add_argument('--features', type=int, default=4)  # 1, 2, 4, 6
    args = parser.parse_args()
    args.models = args.models.split('_')
    wab = WeightedAverageBlender1(args, logging.getLogger(__name__))
    wab.aggregate_scores()
