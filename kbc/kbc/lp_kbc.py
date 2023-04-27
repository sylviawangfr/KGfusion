# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
from typing import Dict
import torch
from pykeen.datasets import get_dataset
from pykeen.utils import resolve_device
from torch import optim

from analysis.group_eval_utils import get_all_pos_triples, to_fusion_eval_format
from kbc.datasets import Dataset
from kbc.models import CP, ComplEx
from kbc.regularizers import F2, N3
from kbc.optimizers import KBCOptimizer
from common_utils import save_to_file


def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    """
    aggregate metrics for missing lhs and rhs
    :param mrrs: d
    :param hits:
    :return:
    """
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    return {'MRR': m, 'hits@[1,3,10]': h}


class LpKBC:
    def __init__(self, params):
        self.params = params
        self.dataset = get_dataset(dataset=params.dataset)
        self.work_dir = params.work_dir

    def train_and_pred(self):
        args = self.params
        kbc_dataset = Dataset(args.dataset)
        examples = torch.from_numpy(kbc_dataset.get_train().astype('int64'))
        print(kbc_dataset.get_shape())
        model = {
            'CP': lambda: CP(kbc_dataset.get_shape(), args.rank, args.init),
            'ComplEx': lambda: ComplEx(kbc_dataset.get_shape(), args.rank, args.init),
        }[args.model]()

        regularizer = {
            'F2': F2(args.reg),
            'N3': N3(args.reg),
        }[args.regularizer]

        device = resolve_device()
        print(f"model device: {str(device)}")
        model.to(device)
        # examples.to(device)

        optim_method = {
            'Adagrad': lambda: optim.Adagrad(model.parameters(), lr=args.learning_rate),
            'Adam': lambda: optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.decay1, args.decay2)),
            'SGD': lambda: optim.SGD(model.parameters(), lr=args.learning_rate)
        }[args.optimizer]()

        optimizer = KBCOptimizer(model, regularizer, optim_method, args.batch_size)
        curve = {'train': [], 'valid': [], 'test': []}
        for e in range(args.max_epochs):
            optimizer.epoch(examples)
            if (e + 1) % args.valid == 0:
                valid, test, train = [
                    avg_both(*kbc_dataset.eval(model, split, -1 if split != 'train' else 50000))
                    for split in ['valid', 'test', 'train']
                ]
                curve['valid'].append(valid)
                curve['test'].append(test)
                curve['train'].append(train)

                print("\t TRAIN: ", train)
                print("\t VALID : ", valid)
        result_test = kbc_dataset.eval(model, 'test', -1)
        str_ht = str(result_test)
        str_both = str(avg_both(result_test[0], result_test[1]))
        save_to_file("{}\n both: {}".format(str_ht, str_both), args.work_dir + "result.txt")
        # ht = dataset.eval(model, 'valid', -1)
        # both = avg_both(ht[0], ht[1])
        # total_hits = torch.stack([ht[1]['lhs'], ht[1]['rhs'], both['hits@[1,3,10]']], 0)
        # total_mrr = torch.as_tensor([ht[0]['lhs'], ht[0]['rhs'], both['MRR']]).unsqueeze(1)
        # total_eval = torch.cat([total_hits, total_mrr], 1)
        # torch.save(total_eval, args.out_dir + "rank_total_eval.pt")
        pykeen_dataset = get_dataset(dataset=args.dataset)
        test_scores = kbc_dataset.pred(model, 'test', -1, filter=False)
        # test_scores = dataset.pred(model, 'test', -1, filter=True)
        # eval_groups(pykeen_dataset.testing.mapped_triples, test_scores)
        torch.save(test_scores, args.work_dir + "preds.pt")
        dev_scores = kbc_dataset.pred(model, 'valid', -1, filter=True)
        torch.save(dev_scores, args.work_dir + "valid_preds.pt")
        all_pos_triples = get_all_pos_triples(pykeen_dataset)
        to_fusion_eval_format(pykeen_dataset, dev_scores,
                              all_pos_triples, args.work_dir, top_k=100)
        # per_rel_eval(pykeen_dataset.validation.mapped_triples, dev_scores, args.out_dir)
        # per_mapping_eval(pykeen_dataset, dev_scores, args.out_dir)


if __name__ == '__main__':
    datasets = ['WN18RR', 'FB15k237', 'UMLS']
    parser = argparse.ArgumentParser(
        description="Relational learning contraption"
    )
    parser.add_argument(
        '--dataset', choices=datasets, default='UMLS',
        help="Dataset in {}".format(datasets)
    )
    models = ['CP', 'ComplEx']
    parser.add_argument(
        '--model', choices=models, default="ComplEx",
        help="Model in {}".format(models)
    )
    regularizers = ['N3', 'F2']
    parser.add_argument(
        '--regularizer', choices=regularizers, default='N3',
        help="Regularizer in {}".format(regularizers)
    )
    optimizers = ['Adagrad', 'Adam', 'SGD']
    parser.add_argument(
        '--optimizer', choices=optimizers, default='Adagrad',
        help="Optimizer in {}".format(optimizers)
    )
    parser.add_argument(
        '--max_epochs', default=50, type=int,
        help="Number of epochs."
    )
    parser.add_argument(
        '--valid', default=10, type=float,
        help="Number of epochs before valid."
    )
    parser.add_argument(
        '--rank', default=500, type=int,
        help="Factorization rank."
    )
    parser.add_argument(
        '--batch_size', default=1000, type=int,
        help="Factorization rank."
    )
    parser.add_argument(
        '--reg', default=0, type=float,
        help="Regularization weight"
    )
    parser.add_argument(
        '--init', default=1e-3, type=float,
        help="Initial scale"
    )
    parser.add_argument(
        '--learning_rate', default=1e-1, type=float,
        help="Learning rate"
    )
    parser.add_argument(
        '--decay1', default=0.9, type=float,
        help="decay rate for the first moment estimate in Adam"
    )
    parser.add_argument(
        '--decay2', default=0.999, type=float,
        help="decay rate for second moment estimate in Adam"
    )
    parser.add_argument(
        '--work_dir', type=str, default="../../outputs/UMLS/ComplEx/"
    )
    m_args = parser.parse_args()
    lpkbc = LpKBC(m_args)
    lpkbc.train_and_pred()
