import argparse
import gc
import logging

import numpy as np
import torch
from netcal.scaling import LogisticCalibration
from pykeen.datasets import get_dataset
from pykeen.evaluation import RankBasedEvaluator
from pykeen.typing import LABEL_HEAD, LABEL_TAIL
from context_load_and_run import load_score_context
from features.feature_scores_only_dataset import ScoresOnlyDataset
from common_utils import format_result, save_to_file
from lp_kge.lp_pykeen import get_all_pos_triples

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


class PlattScalingIndividual():
    def __init__(self, params):
        self.work_dir = params['work_dir']
        self.dataset = get_dataset(
            dataset=params['dataset']
        )
        self.model_list = params['models']
        self.num_neg = params['num_neg']
        self.context = load_score_context(params['models'],
                                          in_dir=params['work_dir']
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
            logistic = LogisticCalibration(method='variational', detection=True, independent_probabilities=True,
                                           use_cuda=use_cuda, vi_epochs=500)
            logistic.fit(m.numpy(), labels)
            # individual_cali = logistic.transform(pred_features[index].numpy(), mean_estimate=True)
            logger.info(f"Start transforming {self.model_list[index]}.")
            individual_cali = logistic.transform(pred_features[index].numpy()).mean(0)
            old_shape = self.context[model_name]['preds'].shape
            h_preds, t_preds = torch.chunk(torch.from_numpy(individual_cali), 2, 0)
            h_preds = torch.reshape(h_preds, (old_shape[0], int(old_shape[1]/2)))
            t_preds = torch.reshape(t_preds, (old_shape[0], int(old_shape[1]/2)))
            individual_cali = torch.cat([h_preds, t_preds], 1)
            torch.save(individual_cali, self.work_dir + f"{model_name}/cali_preds.pt")
            logger.info(f"Transforming saved for {self.model_list[index]}.")
            del logistic
            del h_preds
            del t_preds
            del individual_cali
            gc.collect()
            if use_cuda:
                torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="ComplEx_TuckER")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument("--num_neg", type=int, default=10)
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    args = parser.parse_args()
    param1 = args.__dict__
    param1.update({"models": args.models.split('_')})
    wab = PlattScalingIndividual(param1)
    wab.cali()
