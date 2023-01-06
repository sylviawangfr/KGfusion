import argparse
import torch
from pykeen.datasets import get_dataset
from pykeen.evaluation import RankBasedEvaluator
from pykeen.typing import LABEL_HEAD, LABEL_TAIL
from blenders.blender_utils import restore_eval_format
from common_utils import format_result, save_to_file
from context_load_and_run import load_score_context
from features.feature_per_rel_ent_dataset import PerRelEntDataset
from features.feature_per_rel_both_dataset import PerRelBothDataset
from lp_kge.lp_pykeen import get_all_pos_triples


class WeightedAverageBlender:
    def __init__(self, params):
        self.dataset = get_dataset(
            dataset=params['dataset']
        )
        self.params = params

    def _p1(self, t1, t2):
        return 2 * torch.div(torch.mul(t1, t2), torch.add(t1, t2))

    def _p2(self, t1, t2):
        tmp_shape = t1.shape
        tmp1 = torch.sub(torch.ones(tmp_shape), t1)
        tmp2 = torch.sub(torch.ones(tmp_shape), t2)
        tmp3 = torch.mul(tmp1, tmp2)
        return torch.sub(torch.ones(tmp_shape), tmp3)

    def aggregate_scores(self):
        work_dir = self.params['work_dir']
        context = load_score_context(self.params['models'], in_dir=work_dir)
        mapped_triples = self.dataset.testing.mapped_triples
        all_pos = get_all_pos_triples(self.dataset)
        test_data_feature = PerRelEntDataset(mapped_triples, context, all_pos)
        rel_eval_feature, ent_rel_feature, score_feature = torch.chunk(test_data_feature.get_all_test_examples(), 3, 1)
        eval_balanced = self._p2(rel_eval_feature, ent_rel_feature)
        eval_mul_score = torch.sum(torch.mul(eval_balanced, score_feature), 1)
        tmp_evl_sum = torch.sum(eval_balanced, 1)
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
        str_re = format_result(result)
        save_to_file(str_re, work_dir + '_'.join(self.params['models']) + "_w_avg2.log")
        print(str_re)
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
