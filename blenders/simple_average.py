import argparse
import torch
from pykeen.evaluation import RankBasedEvaluator
from pykeen.typing import LABEL_HEAD, LABEL_TAIL
from blenders.blender_utils import restore_eval_format, Blender
from common_utils import format_result, save_to_file
from context_load_and_run import load_score_context
from features.feature_scores_only_dataset import ScoresOnlyDataset
from lp_kge.lp_pykeen import get_all_pos_triples


class SimpleAverageBlender(Blender):
    def __init__(self, params):
        super().__init__(params)
        self.context = load_score_context(self.params['models'],
                                          in_dir=params['work_dir'],
                                          rel_mapping=False,
                                          calibration=params['cali']=="True"
                                          )

    def aggregate_scores(self):
        work_dir = self.params['work_dir']
        mapped_triples = self.dataset.testing.mapped_triples
        all_pos = get_all_pos_triples(self.dataset)
        test_data_feature = ScoresOnlyDataset(mapped_triples, self.context, all_pos)
        score_feature = test_data_feature.get_all_test_examples()
        blender = torch.mean(score_feature, dim=1)
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
        option_str = f"{self.params['dataset']}_{'_'.join(self.params['models'])}_{self.params['cali']}_" \
                     "simple_avg"
        save_to_file(str_re, work_dir + f"{option_str}.log")
        print(f"{option_str}:\n{str_re}")
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="ComplEx_TuckER")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    parser.add_argument("--cali", type=str, default="True")
    args = parser.parse_args()
    param1 = args.__dict__
    param1.update({"models": args.models.split('_')})
    wab = SimpleAverageBlender(param1)
    wab.aggregate_scores()
