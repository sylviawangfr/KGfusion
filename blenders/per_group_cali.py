import argparse
import torch
from pykeen.evaluation import RankBasedEvaluator
from pykeen.typing import LABEL_HEAD, LABEL_TAIL
from blenders.blender_utils import restore_eval_format, Blender
from common_utils import format_result, save_to_file
from context_load_and_run import load_score_context
from features.feature_scores_only_dataset import ScoresOnlyDataset
from lp_kge.lp_pykeen import get_all_pos_triples
import pandas as pd


class SimpleAverageBlender(Blender):
    def __init__(self, params):
        super().__init__(params)
        self.context = load_score_context(self.params['models'],
                                          in_dir=params['work_dir'],
                                          rel_mapping=params['rel_mapping'],
                                          calibration=False
                                          )

    def aggregate_scores(self):
        # calibrate each group of triples separately
        mapped_triples_eval = self.dataset.validation.mapped_triples
        triples_df = pd.DataFrame(data=mapped_triples_eval.numpy(), columns=['h', 'r', 't'])
        groups = triples_df.groupby('r', group_keys=True, as_index=False)
        # get rel id to eval tensor row index
        releval2idx = {key: idx for idx, key in enumerate(groups.groups.keys())}