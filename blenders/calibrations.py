import argparse
import gc
import logging
import sys
import torch
from netcal.scaling import LogisticCalibration
from torch.utils.data import DataLoader
from tqdm import tqdm
from context_load_and_run import load_score_context
from features.feature_scores_only_dataset import ScoresOnlyDataset
from lp_kge.lp_pykeen import get_all_pos_triples
from blenders.blender_base import Blender
import numpy as np


class CalibrationBlender1(Blender):
    def __init__(self, params, logger):
        super().__init__(params, logger)
        self.context = load_score_context(self.params.models,
                                          in_dir=params.work_dir,
                                          calibration=False
                                          )

    def aggregate_scores(self):
        all_pos_triples = get_all_pos_triples(self.dataset)
        models_context = self.context
        dev_feature_dataset = ScoresOnlyDataset(self.dataset.validation.mapped_triples,
                                                models_context,
                                                all_pos_triples,
                                                num_neg=self.params.num_neg)
        test_feature_dataset = ScoresOnlyDataset(self.dataset.testing.mapped_triples, models_context, all_pos_triples)
        use_cuda = torch.cuda.is_available()
        if self.params.cali == "momentum":
            cali = LogisticCalibration(method='momentum',
                                       detection=True,
                                       independent_probabilities=True,
                                       use_cuda=use_cuda,
                                       momentum_epochs=self.params.epoch,
                                       vi_epochs=self.params.epoch)
        elif self.params.cali == "variational":
            self.logger.info("using variational")
            cali = LogisticCalibration(method='variational',
                                       detection=True,
                                       independent_probabilities=True,
                                       use_cuda=use_cuda,
                                       vi_epochs=self.params.epoch)
        else:
            self.logger.info("unsupported cali function, please set cali in ['variational', 'momentum']")
            sys.exit()
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
        candidate_number = self.dataset.num_entities
        m_test_dataloader = DataLoader(pred_features.numpy(), batch_size=512 * candidate_number)
        individual_cali = []
        for batch in tqdm(m_test_dataloader):
            if self.params.cali == "variational":
                batch_cali = cali.transform(batch.numpy(), num_samples=100).mean(0)
            else:
                batch_cali = cali.transform(batch.numpy())
            individual_cali.extend(batch_cali)
            gc.collect()
            if use_cuda:
                torch.cuda.empty_cache()

        option_str = f"{self.params.dataset}_{'_'.join(self.params.models)}_{self.params.cali}_cali"
        ht_blender = torch.as_tensor(individual_cali)
        self.finalize(ht_blender, option_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="CPComplEx_anyburl")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument("--num_neg", type=int, default=40)
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--cali", type=str, default="variational")
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    args = parser.parse_args()
    args.models = args.models.split('_')
    wab = CalibrationBlender1(args, logging.getLogger(__name__))
    wab.aggregate_scores()
