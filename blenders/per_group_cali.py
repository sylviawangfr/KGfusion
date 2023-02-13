import pandas as pd
import argparse
import gc
import logging
import torch
from netcal.scaling import LogisticCalibration
from pykeen.evaluation import RankBasedEvaluator
from pykeen.typing import LABEL_HEAD, LABEL_TAIL
from torch.utils.data import DataLoader
from tqdm import tqdm

from common_utils import format_result, save_to_file
from context_load_and_run import load_score_context
from features.feature_scores_only_dataset import ScoresOnlyDataset
from lp_kge.lp_pykeen import get_all_pos_triples, find_relation_mappings
from blender_utils import restore_eval_format, Blender


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


class PerGroupCaliBlender(Blender):
    def __init__(self, params):
        super().__init__(params)
        self.rel_mapping = self.params['rel_mapping']=='True'
        self.context = load_score_context(self.params['models'],
                                          in_dir=params['work_dir'],
                                          rel_mapping=params['rel_mapping']=='True',
                                          calibration=False
                                          )

    def _groupby_rel(self):
        eval_triples_df = pd.DataFrame(data=self.dataset.validation.mapped_triples.numpy(), columns=['h', 'r', 't'])
        eval_groups = eval_triples_df.groupby('r', group_keys=True, as_index=False)
        eval_rel2index = {key: group.index.values for key, group in eval_groups}
        test_triples_df = pd.DataFrame(data=self.dataset.testing.mapped_triples.numpy(), columns=['h', 'r', 't'])
        test_groups = test_triples_df.groupby('r', group_keys=True, as_index=False)
        test_rel2index = {key: group.index.values for key, group in test_groups}
        return eval_rel2index, test_rel2index

    def _groupby_mapping(self):
        mappings = find_relation_mappings(self.dataset)
        eval_triples_df = pd.DataFrame(data=self.dataset.validation.mapped_triples.numpy(), columns=['h', 'r', 't'])
        test_triples_df = pd.DataFrame(data=self.dataset.testing.mapped_triples.numpy(), columns=['h', 'r', 't'])
        eval_rel2index = {key: eval_triples_df.query("r in @rels").index.values for key, rels in mappings.items()}
        test_rel2index = {key: test_triples_df.query("r in @rels").index.values for key, rels in mappings.items()}
        return eval_rel2index, test_rel2index

    def aggregate_scores(self):
        # calibrate each group of triples separately
        all_pos_triples = get_all_pos_triples(self.dataset)
        work_dir = self.params['work_dir']
        models_context = self.context
        if self.rel_mapping:
            eval_rel2index, test_rel2index = self._groupby_mapping()
        else:
            eval_rel2index, test_rel2index = self._groupby_rel()
        dev_feature_dataset = ScoresOnlyDataset(self.dataset.validation.mapped_triples,
                                                models_context,
                                                all_pos_triples,
                                                num_neg=self.params['num_neg'])
        test_feature_dataset = ScoresOnlyDataset(self.dataset.testing.mapped_triples, models_context, all_pos_triples)
        calibrated_scores = []
        reorded_index = []
        use_cuda = torch.cuda.is_available()
        logger.debug(f"use cuda: {use_cuda}")
        model_num = len(self.context['models'])
        for key, tri_index in tqdm(eval_rel2index.items()):
            if key not in test_rel2index or len(tri_index) == 0 or len(test_rel2index[key])==0:
                continue
            pos, neg = dev_feature_dataset.collate_train(tri_index)
            combined_feature = torch.cat([pos, neg], 0)
            labels = torch.cat([torch.ones(pos.shape[0], 1),
                                torch.zeros(neg.shape[0], 1)], 0).numpy()
            model_features = torch.chunk(combined_feature, model_num, 1)
            pred_features = test_feature_dataset.collate_test(test_rel2index[key])
            pred_features = torch.chunk(pred_features, model_num, 1)
            reorded_index.extend(test_rel2index[key])
            rel_logits = []
            for index, m in enumerate(model_features):
                logistic = LogisticCalibration(method='variational', detection=True, independent_probabilities=True,
                                               use_cuda=use_cuda, vi_epochs=500)
                logistic.fit(model_features[index].numpy(), labels)
                rel_test_dataloader = DataLoader(pred_features[index].numpy(), batch_size=1000)
                rel_m_cali = []
                for batch in rel_test_dataloader:
                    batch_individual_cali = logistic.transform(batch.numpy(), num_samples=100).mean(0)
                    rel_m_cali.extend(batch_individual_cali)
                gc.collect()
                if use_cuda:
                    torch.cuda.empty_cache()
                rel_logits.append(torch.as_tensor(rel_m_cali))
            rel_logits = torch.vstack(rel_logits)
            rel_h_preds, rel_t_preds = torch.chunk(rel_logits, 2, 1)
            calibrated_scores.append([rel_h_preds, rel_t_preds])
        # order test set as sequence of relation groups
        reorded_index = torch.as_tensor(reorded_index, dtype=torch.int64)
        reorder_test_tris = self.dataset.testing.mapped_triples[reorded_index]
        h_preds = [g[0] for g in calibrated_scores]
        t_preds = [g[1] for g in calibrated_scores]
        # average model scores
        ht_scores = [torch.cat(h_preds, 1).mean(0), torch.cat(t_preds, 1).mean(0)]
        evaluator = RankBasedEvaluator()
        relation_filter = None
        for ind, target in enumerate([LABEL_HEAD, LABEL_TAIL]):
            relation_filter = restore_eval_format(
                batch=reorder_test_tris,
                scores=ht_scores[ind],
                target=target,
                evaluator=evaluator,
                all_pos_triples=all_pos_triples,
                relation_filter=relation_filter,
            )
        result = evaluator.finalize()
        str_re = format_result(result)
        option_str = f"{self.params['dataset']}_{'_'.join(self.params['models'])}_mapping{self.rel_mapping}_group_cali"
        save_to_file(str_re, work_dir + f"{option_str}.log")
        print(f"{option_str}:\n{str_re}")
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="ComplEx_TuckER")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument("--num_neg", type=int, default=10)
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    parser.add_argument('--rel_mapping', type=str, default='False')
    args = parser.parse_args()
    param1 = args.__dict__
    param1.update({"models": args.models.split('_')})
    wab = PerGroupCaliBlender(param1)
    wab.aggregate_scores()




