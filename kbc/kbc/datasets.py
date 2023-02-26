# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
from pathlib import Path
import pkg_resources
import pickle
from typing import Dict, Tuple, List

import numpy as np
import torch


from kbc.models import KBCModel


DATA_PATH = Path(pkg_resources.resource_filename('kbc', 'data/'))


class Dataset(object):
    def __init__(self, name: str):
        self.root = DATA_PATH / name

        self.data = {}
        for f in ['train', 'test', 'valid']:
            in_file = open(str(self.root / (f + '.pickle')), 'rb')
            self.data[f] = pickle.load(in_file)

        maxis = np.max(self.data['train'], axis=0)
        self.n_entities = int(max(maxis[0], maxis[2]) + 1)
        self.n_predicates = int(maxis[1] + 1)
        self.n_predicates *= 2

        inp_f = open(str(self.root / f'to_skip.pickle'), 'rb')
        self.to_skip: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)
        inp_f.close()

    def get_examples(self, split):
        return self.data[split]

    def get_train(self):
        copy = np.copy(self.data['train'])
        tmp = np.copy(copy[:, 0])
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] += self.n_predicates // 2  # has been multiplied by two.
        return np.vstack((self.data['train'], copy))

    def eval(
            self, model: KBCModel, split: str, n_queries: int = -1, missing_eval: str = 'both',
            at: Tuple[int] = (1, 3, 10)
    ):
        test = self.get_examples(split)
        examples = torch.from_numpy(test.astype('int64'))
        if torch.cuda.is_available():
            examples.to('cuda')
        missing = [missing_eval]
        if missing_eval == 'both':
            missing = ['rhs', 'lhs']

        mean_reciprocal_rank = {}
        hits_at = {}

        for m in missing:
            q = examples.clone()
            if n_queries > 0:
                permutation = torch.randperm(len(examples))[:n_queries]
                q = examples[permutation]
            if m == 'lhs':
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.n_predicates // 2
            ranks = model.get_ranking(q, self.to_skip[m], batch_size=500)
            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            hits_at[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks <= x).float()).item(),
                at
            ))))

        return mean_reciprocal_rank, hits_at

    def pred(
            self, model: KBCModel, split: str, n_queries: int = -1, missing_eval: str = 'both', filter=False
    ):
        test = self.get_examples(split)
        examples = torch.from_numpy(test.astype('int64'))
        if torch.cuda.is_available():
            examples.to('cuda')
        missing = [missing_eval]
        if missing_eval == 'both':
            missing = ['rhs', 'lhs']
        kbc2pykeen, _ = get_dicts(self.root)
        h_t_preds = []
        for m in missing:
            q = examples.clone()
            if n_queries > 0:
                permutation = torch.randperm(len(examples))[:n_queries]
                q = examples[permutation]
            if m == 'lhs':
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.n_predicates // 2
            # q to original id and order
            if filter:
                scores = model.get_preds(q, self.to_skip[m])
            else:
                scores = model.get_preds(q)
            reordered_scores = kbc_to_pykeen_scores(scores.detach().cpu(), kbc2pykeen)
            h_t_preds.append(reordered_scores)
        h_t_preds.reverse()
        ht = torch.cat(h_t_preds, 1)
        return ht

    def get_shape(self):
        return self.n_entities, self.n_predicates, self.n_entities


def get_dicts(data_dir):
    entities_to_id = dict()
    relations_to_id = dict()
    for (dic, f) in zip([entities_to_id, relations_to_id], ['ent_id', 'rel_id']):
        ff = open(os.path.join(data_dir, f), 'r')
        for line in ff.readlines():
            kbc, pyk = line.strip().split('\t')
            dic.update({int(kbc): int(pyk)})
        ff.close()
    return entities_to_id, relations_to_id


def kbc_to_pykeen_scores(kbc_preds, kbc2pykeen):
    kbcids = list(kbc2pykeen.keys())
    kbcids.sort()
    pykeen_orded_index = [kbc2pykeen[k] for k in kbcids]
    reorded = kbc_preds[:, torch.as_tensor(pykeen_orded_index)]
    return reorded
