import argparse
import gc
import logging
import torch
from netcal.binning import IsotonicRegression
from netcal.scaling import LogisticCalibration
from pykeen.evaluation import RankBasedEvaluator
from pykeen.typing import LABEL_HEAD, LABEL_TAIL
from torch.utils.data import DataLoader
from tqdm import tqdm

from context_load_and_run import load_score_context
from features.feature_scores_only_dataset import ScoresOnlyDataset
from lp_kge.lp_pykeen import get_all_pos_triples
from blender_utils import eval_with_blender_scores, Blender
from common_utils import format_result, save_to_file

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


class CalibrationBlender2(Blender):
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
        # detection : bool, default: False
        #     If True, the input array 'X' is treated as a box predictions with several box features (at least
        # box confidence must be present) with shape (n_samples, [n_box_features]).
        pos, neg = dev_feature_dataset.get_all_dev_examples()
        inputs = torch.cat([pos, neg], 0)
        labels = torch.cat([torch.ones(pos.shape[0], 1),
                            torch.zeros(neg.shape[0], 1)], 0).numpy()
        model_num = inputs.shape[1]
        model_features = torch.chunk(inputs, model_num, 1)
        use_cuda = torch.cuda.is_available()
        logger.debug(f"use cuda: {use_cuda}")
        pred_features = test_feature_dataset.get_all_test_examples()
        pred_features = torch.chunk(pred_features, model_num, 1)
        ens_logits = []
        for index, m in enumerate(model_features):
            model_name = self.context['models'][index]
            old_shape = self.context[model_name]['preds'].shape
            if self.params.cali == "scaling":
                cali = LogisticCalibration(method='variational', detection=True, independent_probabilities=True,
                                           use_cuda=use_cuda, vi_epochs=500)
            else:
                cali = IsotonicRegression(detection=True, independent_probabilities=True)
            cali.fit(m.numpy(), labels)
            logger.info(f"Start transforming {model_name}.")
            m_test_dataloader = DataLoader(pred_features[index].numpy(), batch_size=100 * old_shape[1])
            individual_cali = []
            for batch in tqdm(m_test_dataloader):
                batch_individual_cali = cali.transform(batch.numpy())
                if self.params.cali == "scaling":
                    batch_individual_cali = batch_individual_cali.mean(0)
                individual_cali.extend(batch_individual_cali)
                gc.collect()
                if use_cuda:
                    torch.cuda.empty_cache()
            ens_logits.append(torch.as_tensor(individual_cali))
        ens_logits = torch.mean(torch.vstack(ens_logits), 0)
        h_preds, t_preds = torch.chunk(ens_logits, 2, 0)
        # restore format that required by pykeen evaluator
        ht_scores = [h_preds, t_preds]
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
        option_str = f"{self.params.dataset}_{'_'.join(self.params.models)}_ScallinCali2"
        save_to_file(str_re, work_dir + f"{option_str}.log")
        print(f"{option_str}:\n{str_re}")
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="ComplEx_TuckER_RotatE")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument("--num_neg", type=int, default=10)
    parser.add_argument("--cali", type=str, default="scaling")
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    args = parser.parse_args()
    args.models = args.models.split('_')
    wab = CalibrationBlender2(args)
    wab.aggregate_scores()
