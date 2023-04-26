import argparse
import logging
import torch
from blenders.blender_utils import Blender
from context_load_and_run import load_score_context
from features.feature_per_rel_ent_dataset import PerRelEntDataset
from lp_kge.lp_pykeen import get_all_pos_triples


class WeightedAverageBlender2(Blender):
    def __init__(self, params, logger):
        super().__init__(params, logger)
        self.context = load_score_context(self.params.models,
                                          in_dir=params.work_dir,
                                          evaluator_key=params.evaluator_key,
                                          eval_feature=params.evaluator_feature,
                                          )

    def _p1(self, t1, t2):
        return 2 * torch.div(torch.mul(t1, t2), torch.add(t1, t2))

    def _p2(self, t1, t2):
        tmp_shape = t1.shape
        tmp1 = torch.sub(torch.ones(tmp_shape), t1)
        tmp2 = torch.sub(torch.ones(tmp_shape), t2)
        tmp3 = torch.mul(tmp1, tmp2)
        return torch.sub(torch.ones(tmp_shape), tmp3)

    def aggregate_scores(self):
        mapped_triples = self.dataset.testing.mapped_triples
        all_pos = get_all_pos_triples(self.dataset)
        test_data_feature = PerRelEntDataset(mapped_triples, self.context, all_pos)
        rel_eval_feature, ent_rel_feature, score_feature = torch.chunk(test_data_feature.get_all_test_examples(), 3, 1)
        eval_balanced = self._p2(rel_eval_feature, ent_rel_feature)
        eval_mul_score = torch.sum(torch.mul(eval_balanced, score_feature), 1)
        tmp_evl_sum = torch.sum(eval_balanced, 1)
        tmp_evl_sum[tmp_evl_sum == 0] = 0.5  # do not divided by zero
        blender_score = torch.div(eval_mul_score, tmp_evl_sum)
        option_str = f"{self.params.dataset}_{'_'.join(self.params.models)}_" \
                     f"{self.params.evaluator_key}_wavg2"
        self.finalize(blender_score, option_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="ComplEx_TuckER")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    parser.add_argument('--evaluator_key', type=str, default="rank")
    args = parser.parse_args()
    args.models = args.models.split('_')
    wab = WeightedAverageBlender2(args, logging.getLogger(__name__))
    wab.aggregate_scores()
