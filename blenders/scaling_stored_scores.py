import argparse
import gc
import logging
import sys

import torch
from netcal.binning import IsotonicRegression
from netcal.scaling import LogisticCalibration, TemperatureScaling
from pykeen.datasets import get_dataset
from pykeen.evaluation import RankBasedEvaluator
from pykeen.typing import LABEL_HEAD, LABEL_TAIL
from torch.utils.data import DataLoader
from tqdm import tqdm

from blenders.blender_utils import eval_with_blender_scores
from common_utils import format_result
from context_load_and_run import load_score_context
from features.feature_scores_only_dataset import ScoresOnlyDataset
from lp_kge.lp_pykeen import get_all_pos_triples

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


class PlattScalingIndividual():
    def __init__(self, params):
        self.work_dir = params.work_dir
        self.dataset = get_dataset(
            dataset=params.dataset
        )
        self.model_list = params.models
        self.num_neg = params.num_neg
        self.params = params
        self.context = load_score_context(params.models,
                                          in_dir=params.work_dir,
                                          calibration=False
                                          )

    def cali(self):
        all_pos_triples = get_all_pos_triples(self.dataset)
        models_context = self.context
        dev_feature_dataset = ScoresOnlyDataset(self.dataset.validation.mapped_triples,
                                                models_context,
                                                all_pos_triples,
                                                num_neg=self.num_neg)
        test_feature_dataset = ScoresOnlyDataset(self.dataset.testing.mapped_triples, models_context, all_pos_triples)
        # detection : bool, default: False
        #     If True, the input array 'X' is treated as a box predictions with several box features (at least
        # box confidence must be present) with shape (n_samples, [n_box_features]).
        pos, neg = dev_feature_dataset.get_all_dev_examples()
        keep_index1 = (pos > 0).nonzero(as_tuple=True)[0]
        neg = neg.reshape(pos.shape[0], self.num_neg)[keep_index1].flatten().unsqueeze(1)
        pos = pos[keep_index1]
        keep_index2 = (neg > 0).nonzero(as_tuple=True)[0]
        neg = neg[keep_index2]
        logger.info(f"pos num: {pos.shape[0]}")
        logger.info(f"neg num: {neg.shape[0]}")
        inputs = torch.cat([pos, neg], 0)
        labels = torch.cat([torch.ones(pos.shape[0], 1),
                            torch.zeros(neg.shape[0], 1)], 0).numpy()
        model_num = inputs.shape[1]
        model_features = torch.chunk(inputs, model_num, 1)
        use_cuda = torch.cuda.is_available()
        logger.debug(f"use cuda: {use_cuda}")
        pred_features = test_feature_dataset.get_all_test_examples()
        pred_features = torch.chunk(pred_features, model_num, 1)

        for index, m in enumerate(model_features):
            model_name = self.model_list[index]
            if self.params.cali == "variational":
                logger.info("using variational")
                cali = LogisticCalibration(method='variational',
                                           detection=True,
                                           independent_probabilities=True,
                                           use_cuda=use_cuda,
                                           vi_epochs=500)
            elif self.params.cali == 'isotonic':
                logger.info("using isotonic")
                cali = IsotonicRegression(detection=True, independent_probabilities=True)
            elif self.params.cali == 'momentum':
                logger.info("using momentum")
                cali = LogisticCalibration(method='momentum',
                                           detection=True,
                                           independent_probabilities=True,
                                           momentum_epochs=500,
                                           vi_epochs=500)
            else:
                logger.info("unsupported cali function, please set cali in ['variational', 'isotonic', 'momentum']")
                sys.exit()
            gc.collect()
            if use_cuda:
                torch.cuda.empty_cache()
            cali.fit(m.numpy(), labels)
            old_shape = self.context[model_name]['preds'].shape
            logger.info(f"Start transforming {self.model_list[index]}.")
            m_test_dataloader = DataLoader(pred_features[index].numpy(), batch_size=512 * old_shape[1])
            individual_cali = []
            for batch in tqdm(m_test_dataloader):
                if self.params.cali == "variational":
                    batch_individual_cali = cali.transform(batch.numpy(), num_samples=100).mean(0)
                else:
                    batch_individual_cali = cali.transform(batch.numpy())
                individual_cali.extend(batch_individual_cali)
                gc.collect()
                if use_cuda:
                    torch.cuda.empty_cache()

            h_preds, t_preds = torch.chunk(torch.as_tensor(individual_cali), 2, 0)
            h_preds = torch.reshape(h_preds, (old_shape[0], int(old_shape[1] / 2)))
            t_preds = torch.reshape(t_preds, (old_shape[0], int(old_shape[1] / 2)))
            individual_cali = torch.cat([h_preds, t_preds], 1)
            torch.save(individual_cali, self.work_dir + f"{model_name}/cali_preds.pt")
            logger.info(f"Transforming saved for {self.model_list[index]}.")
            raw_res = self.test_pred(self.context[model_name]['preds'])
            logger.info(f"{self.model_list[index]} rank evaluation with raw scores:\n {raw_res}")
            cali_res = self.test_pred(individual_cali)
            logger.info(f"{self.model_list[index]} rank evaluation with calibrated scores:\n {cali_res}")
            del cali
            del h_preds
            del t_preds
            del individual_cali
            gc.collect()
            if use_cuda:
                torch.cuda.empty_cache()

    def test_pred(self, pred_scores_ht):
        all_pos = get_all_pos_triples(self.dataset)
        ht_scores = torch.chunk(pred_scores_ht, 2, 1)
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
        return str_re


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="CPComplEx")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument("--num_neg", type=int, default=10)
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    parser.add_argument('--cali', type=str, default="variational")
    args = parser.parse_args()
    args.models = args.models.split('_')
    wab = PlattScalingIndividual(args)
    wab.cali()
