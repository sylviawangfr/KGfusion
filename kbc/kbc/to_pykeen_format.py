# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import pandas as pd
import argparse
from typing import Dict
import pykeen
import torch
from pykeen.datasets import get_dataset
from torch import optim
from kbc.datasets import Dataset
from kbc.models import CP, ComplEx
from kbc.regularizers import F2, N3
from kbc.optimizers import KBCOptimizer
from learn import avg_both
from lp_kge.lp_pykeen import get_neg_scores_top_k, find_relation_mappings, get_all_pos_triples
from lp_rules.lp_anyburl import calc_hit_at_10
from utils import save2json, does_exist


def train_and_pred(args):
    dataset = Dataset(args.dataset)
    examples = torch.from_numpy(dataset.get_train().astype('int64'))
    print(dataset.get_shape())
    model = {
        'CP': lambda: CP(dataset.get_shape(), args.rank, args.init),
        'ComplEx': lambda: ComplEx(dataset.get_shape(), args.rank, args.init),
    }[args.model]()

    regularizer = {
        'F2': F2(args.reg),
        'N3': N3(args.reg),
    }[args.regularizer]

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)

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
                avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
                for split in ['valid', 'test', 'train']
            ]
            curve['valid'].append(valid)
            curve['test'].append(test)
            curve['train'].append(train)

            print("\t TRAIN: ", train)
            print("\t VALID : ", valid)
    result_test = dataset.eval(model, 'test', -1)
    print(result_test)
    results1 = dataset.pred(model, 'test', -1)
    pykeen_dataset = get_dataset(args.dataset)
    all_pos_triples = get_all_pos_triples(pykeen_dataset)
    test_scores = to_fusion_eval_format(pykeen_dataset.testing.mapped_triples, results1, all_pos_triples, args.out_dir)
    torch.save(test_scores, args.out_dir + "preds.pt")
    result2 = dataset.pred(model, 'valid', -1)
    dev_scores = to_fusion_eval_format(pykeen_dataset.validation.mapped_triples, result2, all_pos_triples, args.out_dir)
    per_rel_eval(pykeen_dataset.validation.mapped_triples, dev_scores, args.out_dir)
    per_mapping_eval(pykeen_dataset.validation.mapped_triples, dev_scores, args.out_dir)


def to_fusion_eval_format(mapped_triples, pred_scores, all_pos_triples, out_dir, top_k=10):
    m_dev_preds = torch.chunk(pred_scores, 2, 1)
    m_dev_preds = [i.squeeze(1) for i in m_dev_preds]
    pos_scores = m_dev_preds[0]
    pos_scores = pos_scores[torch.arange(0, mapped_triples.shape[0]),
                            mapped_triples[:, 0]]
    neg_scores, neg_index_topk = get_neg_scores_top_k(mapped_triples, m_dev_preds, all_pos_triples, top_k) # [[h1 * candidate, h2 * candicate...][t1,t2...]]
    torch.save(pos_scores, out_dir + "eval_pos_scores.pt")
    torch.save(neg_scores, out_dir + "eval_neg_scores.pt")
    torch.save(neg_index_topk, out_dir + "eval_neg_index.pt")


def per_rel_eval(mapped_triples, scores, out_dir):
    triples_df = pd.DataFrame(data=mapped_triples.numpy(), columns=['h', 'r', 't'])
    original_groups = triples_df.groupby('r', group_keys=True, as_index=False)
    group_keys = original_groups.groups.keys()
    file_name = "rank_rel_eval.pt"
    head_tail_eval = []
    for rel in group_keys:
    # generate grouped index of eval mapped triples
        g = original_groups.get_group(rel)
        g_index = torch.from_numpy(g.index.values)
        rg_index = torch.as_tensor(g_index)
        h_preds, t_preds = scores[rg_index].chunk(2, 1)
        h_preds = h_preds.squeeze(1)
        t_preds = t_preds.squeeze(1)
        heads = torch.as_tensor(triples_df['h'].values).unsqueeze(1)
        tails = torch.as_tensor(triples_df['t'].values).unsqueeze(1)
        head_hits = calc_hit_at_10(h_preds, heads)
        tail_hits = calc_hit_at_10(t_preds, tails)
        both_hit = calc_hit_at_10(torch.cat([h_preds, t_preds]), torch.cat((heads, tails)))
        head_tail_eval.append([head_hits,
                              tail_hits,
                              both_hit])
    head_tail_eval = torch.Tensor(head_tail_eval)
    torch.save(head_tail_eval, out_dir + file_name)


def per_mapping_eval(pykeen_dataset, scores, out_dir):
    mappings = ['one_to_one', 'one_to_many', 'many_to_one', 'many_to_many']
    rel_mappings = find_relation_mappings(pykeen_dataset)
    dev = pykeen_dataset.validation.mapped_triples
    triples_df = pd.DataFrame(data=dev.numpy(), columns=['h', 'r', 't'])
    relmapping2idx = dict()
    for idx, mapping_type in enumerate(mappings):
        mapped_rels = rel_mappings[mapping_type]
        relmapping2idx.update({int(rel): idx for rel in mapped_rels})
    head_tail_eval = []
    for rel_group in mappings:
        tmp_rels = rel_mappings[rel_group]
        tri_group = triples_df.query('r in @tmp_rels')
        if len(tri_group.index) > 0:
            g = torch.from_numpy(tri_group.values)
            g_index = torch.from_numpy(g.index.values)
            rg_index = torch.as_tensor(g_index)
            h_preds, t_preds = scores[rg_index].chunk(2, 1)
            h_preds = h_preds.squeeze(1)
            t_preds = t_preds.squeeze(1)
            heads = torch.as_tensor(triples_df['h'].values).unsqueeze(1)
            tails = torch.as_tensor(triples_df['t'].values).unsqueeze(1)
            head_hits = calc_hit_at_10(h_preds, heads)
            tail_hits = calc_hit_at_10(t_preds, tails)
            both_hit = calc_hit_at_10(torch.cat([h_preds, t_preds]), torch.cat((heads, tails)))
            head_tail_eval.append([head_hits,
                                   tail_hits,
                                   both_hit])
        else:
            head_tail_eval.append([0.01, 0.01, 0.01])
        torch.save(torch.Tensor(head_tail_eval), out_dir + f"rank_mapping_rel_eval.pt")
    if not does_exist(out_dir + f"rank_mapping_releval2idx.json"):
        save2json(relmapping2idx, out_dir + f"rank_mapping_releval2idx.json")



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
        '--max_epochs', default=5, type=int,
        help="Number of epochs."
    )
    parser.add_argument(
        '--valid', default=5, type=float,
        help="Number of epochs before valid."
    )
    parser.add_argument(
        '--rank', default=50, type=int,
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
        '--out_dir', type=str, default="data/UMLS/"
    )
    m_args = parser.parse_args()
    train_and_pred(m_args)
