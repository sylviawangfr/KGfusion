import argparse
import torch
from pykeen.datasets import get_dataset
from pykeen.evaluation import RankBasedEvaluator
from pykeen.typing import LABEL_HEAD, LABEL_TAIL
from blenders.blender_utils import restore_eval_format, get_blender_dataset
from common_utils import format_result, save_to_file
from context_load_and_run import load_score_context
from features.feature_per_ent_dataset import PerEntDataset
from features.feature_per_rel_dataset import PerRelDataset
from features.feature_per_rel_both_dataset import PerRelBothDataset
from lp_kge.lp_pykeen import get_all_pos_triples
import numpy as np

def weighted_mean(t1, weights):
    eval_mul_score = torch.sum(torch.mul(t1, weights), 1)
    tmp_evl_sum = torch.sum(weights, 1)
    tmp_evl_sum[tmp_evl_sum == 0] = 0.5  # do not divided by zero
    tmp_blender = torch.div(eval_mul_score, tmp_evl_sum)
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


class WeightedAverageBlender:
    def __init__(self, params):
        self.dataset = get_dataset(
            dataset=params['dataset']
        )
        self.params = params

    def aggregate_scores(self):
        work_dir = self.params['work_dir']
        context = load_score_context(self.params['models'], in_dir=work_dir)
        mapped_triples = self.dataset.testing.mapped_triples
        all_pos = get_all_pos_triples(self.dataset)
        get_sampler = get_blender_dataset(param1['sampler'])
        test_data_feature = get_sampler(mapped_triples, context, all_pos)
        eval_feature, score_feature = torch.chunk(test_data_feature.get_all_test_examples(), 2, 1)
        blender = weighted_mean(score_feature, eval_feature)
        h_preds, t_preds = torch.chunk(blender, 2, 0)
        # restore format that required by pykeen evaluator
        ht_scores = [h_preds, t_preds]
        evaluator = RankBasedEvaluator()
        relation_filter = None
        for ind, target in enumerate([LABEL_HEAD, LABEL_TAIL]):
            relation_filter = restore_eval_format(
                batch=self.dataset.testing.mapped_triples,
                scores=ht_scores[ind],
                target=target,
                evaluator=evaluator,
                all_pos_triples=all_pos,
                relation_filter=relation_filter,
            )
        result = evaluator.finalize()
        str_re = format_result(result)
        save_to_file(str_re, work_dir + '_'.join(self.params['models']) + "_w_avg.log")
        print(str_re)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="ComplEx_TuckER")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    # "1": PerRelDataset,
    # "2": PerRelBothDataset,
    # "3": ScoresOnlyDataset,
    # "4": PerEntDataset,
    # "5": PerRelEntDataset
    parser.add_argument('--sampler', type=int, default=4)
    args = parser.parse_args()
    param1 = args.__dict__
    param1.update({"models": args.models.split('_')})
    wab = WeightedAverageBlender(param1)
    wab.aggregate_scores()
