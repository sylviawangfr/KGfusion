import argparse
import torch
from pykeen.evaluation import RankBasedEvaluator
from pykeen.typing import LABEL_HEAD, LABEL_TAIL
from blenders.blender_utils import eval_with_blender_scores, get_features_clz, Blender
from common_utils import format_result, save_to_file
from context_load_and_run import load_score_context
from lp_kge.lp_pykeen import get_all_pos_triples


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
    def __init__(self, params):
        super().__init__(params)
        self.context = load_score_context(self.params.models,
                                          in_dir=params.work_dir,
                                          evaluator_key=params.evaluator_key,
                                          eval_feature=params.eval_feature,
                                          calibration=True
                                          )

    def aggregate_scores(self):
        work_dir = self.params.work_dir
        mapped_triples = self.dataset.testing.mapped_triples
        all_pos = get_all_pos_triples(self.dataset)
        get_features = get_features_clz(self.params.features)
        test_data_feature = get_features(mapped_triples, self.context, all_pos)
        eval_feature, score_feature = torch.chunk(test_data_feature.get_all_test_examples(), 2, 1)
        blender = weighted_mean(score_feature, eval_feature)
        h_preds, t_preds = torch.chunk(blender, 2, 0)
        # restore format that required by pykeen evaluator
        ht_scores = [h_preds, t_preds]
        evaluator = RankBasedEvaluator()
        relation_filter = None
        for ind, target in enumerate([LABEL_HEAD, LABEL_TAIL]):
            relation_filter = eval_with_blender_scores(
                batch=self.dataset.testing.mapped_triples,
                scores=ht_scores[ind],
                target=target,
                evaluator=evaluator,
                all_pos_triples=all_pos,
                relation_filter=relation_filter,
            )
        result = evaluator.finalize()
        str_re = format_result(result)
        option_str = f"{self.params.dataset}_{'_'.join(self.params.models)}_" \
                     f"{self.params.evaluator_key}" \
                     f"evalFeature_{self.params.eval_feature}" \
                     f"data{self.params.features}" \
                     f"_weighted_avg1"
        save_to_file(str_re, work_dir + f"{option_str}.log")
        print(f"{option_str}:\n{str_re}")
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="CPComplEx_CP")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    parser.add_argument('--evaluator_key', type=str, default="rank")
    parser.add_argument('--eval_feature', type=str, default='rel')
    # "1": PerRelDataset,
    # "2": PerRelBothDataset,
    # "3": ScoresOnlyDataset,
    # "4": PerEntDataset,
    # "5": PerRelEntDataset
    # "6": PerModelBothDataset
    parser.add_argument('--features', type=int, default=1)  # 1, 2, 4, 6
    args = parser.parse_args()
    args.models = args.models.split('_')
    wab = WeightedAverageBlender1(args)
    wab.aggregate_scores()
