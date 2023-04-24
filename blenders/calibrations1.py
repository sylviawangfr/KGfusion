import argparse
import logging
import torch
from netcal.binning import IsotonicRegression
from netcal.scaling import LogisticCalibration
from pykeen.evaluation import RankBasedEvaluator
from pykeen.typing import LABEL_HEAD, LABEL_TAIL
from context_load_and_run import load_score_context
from features.feature_scores_only_dataset import ScoresOnlyDataset
from lp_kge.lp_pykeen import get_all_pos_triples
from blender_utils import eval_with_blender_scores, Blender
from common_utils import format_result, save_to_file
import numpy as np

logger = logging.getLogger(__name__)


class CalibrationBlender1(Blender):
    def __init__(self, params):
        super().__init__(params)
        self.context = load_score_context(self.params.models,
                                          in_dir=params.work_dir,
                                          calibration=False
                                          )

    def aggregate_scores(self):
        all_pos_triples = get_all_pos_triples(self.dataset)
        work_dir = self.params.work_dir
        models_context = self.context
        dev_feature_dataset = ScoresOnlyDataset(self.dataset.validation.mapped_triples,
                                                models_context,
                                                all_pos_triples,
                                                num_neg=self.params.num_neg)
        test_feature_dataset = ScoresOnlyDataset(self.dataset.testing.mapped_triples, models_context, all_pos_triples)
        if self.params.cali == "scaling":
            cali = LogisticCalibration(method='momentum',
                                       detection=True,
                                       independent_probabilities=True,
                                       momentum_epochs=500,
                                       vi_epochs=500)
        else:
            cali = IsotonicRegression(detection=True, independent_probabilities=True)
        # detection : bool, default: False
        #     If True, the input array 'X' is treated as a box predictions with several box features (at least
        # box confidence must be present) with shape (n_samples, [n_box_features]).
        pos, neg = dev_feature_dataset.get_all_dev_examples()
        remove_index1 = (pos == 0).nonzero(as_tuple=True)[0]
        keep_index1 = np.delete(np.arange(pos.shape[0]), remove_index1.numpy(), 0)
        neg = neg.reshape(pos.shape[0], int(neg.shape[0]/pos.shape[0]), neg.shape[-1])[keep_index1]
        neg = neg.reshape(neg.shape[0] * neg.shape[1], neg.shape[-1])
        pos = pos[keep_index1]
        remove_index2 = (neg == 0).nonzero(as_tuple=True)[0]
        keep_index2 = np.delete(np.arange(neg.shape[0]), remove_index2.numpy(), 0)
        neg = neg[keep_index2]
        inputs = torch.cat([pos, neg], 0).numpy()
        labels = torch.cat([torch.ones(pos.shape[0], 1),
                            torch.zeros(neg.shape[0], 1)], 0).numpy()
        cali.fit(inputs, labels)
        pred_features = test_feature_dataset.get_all_test_examples()
        ens_logits = cali.transform(pred_features)
        h_preds, t_preds = torch.chunk(torch.as_tensor(ens_logits), 2, 0)
        # restore format that required by pykeen evaluator
        candidate_number = self.dataset.num_entities
        ht_scores = [h_preds.reshape([self.dataset.testing.num_triples, candidate_number]),
                     t_preds.reshape([self.dataset.testing.num_triples, candidate_number])]
        evaluator = RankBasedEvaluator()
        relation_filter = None
        for ind, target in enumerate([LABEL_HEAD, LABEL_TAIL]):
            relation_filter = eval_with_blender_scores(
                batch=test_feature_dataset.mapped_triples,
                scores=ht_scores[ind],
                target=target,
                evaluator=evaluator,
                all_pos_triples=all_pos_triples,
                relation_filter=relation_filter,
            )
        result = evaluator.finalize()
        str_re = format_result(result)
        option_str = f"{self.params.dataset}_{'_'.join(self.params.models)}_Cali1{self.params.cali}"
        save_to_file(str_re, self.log_dir + f"{option_str}.log")
        print(f"{option_str}:\n{str_re}")
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="CP")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument("--num_neg", type=int, default=4)
    parser.add_argument("--cali", type=str, default="scaling")
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    args = parser.parse_args()
    args.models = args.models.split('_')
    wab = CalibrationBlender1(args)
    wab.aggregate_scores()
