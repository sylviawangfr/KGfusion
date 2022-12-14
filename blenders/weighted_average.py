import argparse
import torch
from pykeen.datasets import get_dataset
from pykeen.evaluation import RankBasedEvaluator
from pykeen.typing import LABEL_HEAD, LABEL_TAIL
from blenders.blender_utils import restore_eval_format
from common_utils import format_result
from context_load_and_run import load_score_context
from features.feature_per_rel_ht2_dataset import PerRelNoSignalDataset
from lp_pykeen.lp_pykeen import get_all_pos_triples


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
        test_data_feature = PerRelNoSignalDataset(mapped_triples, context, all_pos)
        eval_feature, score_feature = torch.chunk(test_data_feature.get_all_test_examples(), 2, 1)
        eval_mul_score = torch.sum(torch.mul(eval_feature, score_feature), 1)
        tmp_evl_sum = torch.sum(eval_feature, 1)
        tmp_evl_sum[tmp_evl_sum == 0] = 0.5  # do not divided by zero
        blender = torch.div(eval_mul_score, tmp_evl_sum)
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
        print(format_result(result))
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="ComplEx_TuckER")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    args = parser.parse_args()
    param1 = args.__dict__
    param1.update({"models": args.models.split('_')})
    wab = WeightedAverageBlender(param1)
    wab.aggregate_scores()
