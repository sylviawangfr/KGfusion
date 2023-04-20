import argparse
import torch
from pykeen.evaluation import RankBasedEvaluator
from pykeen.typing import LABEL_HEAD, LABEL_TAIL
from pykeen.utils import resolve_device

from blenders.blender_utils import eval_with_blender_scores, Blender
from common_utils import format_result, save_to_file
from context_load_and_run import load_score_context
from features.feature_scores_only_dataset import ScoresOnlyDataset
from lp_kge.lp_pykeen import get_all_pos_triples, predict_head_tail_scores


class TNormBlender(Blender):
    def __init__(self, params):
        super().__init__(params)
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
        blender = torch.min(score_feature, dim=1).values
        h_preds, t_preds = torch.chunk(blender, 2, 0)
        # restore format that required by pykeen evaluator
        candidate_number = self.dataset.num_entities
        ht_scores = [h_preds.reshape([self.dataset.testing.num_triples, candidate_number]),
                     t_preds.reshape([self.dataset.testing.num_triples, candidate_number])]
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
                     "simple_avg"
        save_to_file(str_re, work_dir + f"{option_str}.log")
        print(f"{option_str}:\n{str_re}")
        return result

    def test_pred(self):
        all_pos = get_all_pos_triples(self.dataset)
        test_data_feature = ScoresOnlyDataset(self.dataset.testing.mapped_triples, self.context, all_pos)
        score_feature = test_data_feature.get_all_test_examples()
        score_feature = torch.chunk(score_feature, len(self.context['models']), 1)
        for idx, m in enumerate(self.params.models):
            ht_scores = torch.chunk(score_feature[idx], 2, 0)
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
            print(f"{m}:\n {str_re}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="ComplEx_RotatE_TuckER_CPComplEx_anyburl")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    args = parser.parse_args()
    args.models = args.models.split('_')
    wab = TNormBlender(args)
    wab.aggregate_scores()
