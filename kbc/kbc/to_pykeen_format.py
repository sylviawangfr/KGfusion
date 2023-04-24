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
from pykeen.utils import resolve_device
from torch import optim
from kbc.datasets import Dataset
from kbc.models import CP, ComplEx
from kbc.regularizers import F2, N3
from kbc.optimizers import KBCOptimizer
from lp_kge.lp_pykeen import get_neg_scores_top_k, find_relation_mappings, get_all_pos_triples
from common_utils import save2json, does_exist, save_to_file


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
                avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
                for split in ['valid', 'test', 'train']
            ]
            curve['valid'].append(valid)
            curve['test'].append(test)
            curve['train'].append(train)

            print("\t TRAIN: ", train)
            print("\t VALID : ", valid)
    result_test = dataset.eval(model, 'test', -1)
    str_ht = str(result_test)
    str_both = str(avg_both(result_test[0], result_test[1]))
    save_to_file("{}\n both: {}".format(str_ht, str_both), args.out_dir + "result.txt")
    ht = dataset.eval(model, 'valid', -1)
    both = avg_both(ht[0], ht[1])
    total_hits = torch.stack([ht[1]['lhs'], ht[1]['rhs'], both['hits@[1,3,10]']], 0)
    total_mrr = torch.as_tensor([ht[0]['lhs'], ht[0]['rhs'], both['MRR']]).unsqueeze(1)
    total_eval = torch.cat([total_hits, total_mrr], 1)
    torch.save(total_eval, args.out_dir + "rank_total_eval.pt")
    pykeen_dataset = get_dataset(dataset=args.dataset)
    test_scores = dataset.pred(model, 'test', -1, filter=False)
    # test_scores = dataset.pred(model, 'test', -1, filter=True)
    # eval_groups(pykeen_dataset.testing.mapped_triples, test_scores)
    torch.save(test_scores, args.out_dir + "preds.pt")
    dev_scores = dataset.pred(model, 'valid', -1, filter=True)
    all_pos_triples = get_all_pos_triples(pykeen_dataset)
    to_fusion_eval_format_and_save_topk(pykeen_dataset.validation.mapped_triples, dev_scores.clone(), all_pos_triples, args.out_dir, top_k=100)
    per_rel_eval(pykeen_dataset.validation.mapped_triples, dev_scores, args.out_dir)
    per_mapping_eval(pykeen_dataset, dev_scores, args.out_dir)


def to_fusion_eval_format_and_save_topk(mapped_triples, pred_scores, all_pos_triples, out_dir, top_k):
    m_dev_preds = torch.chunk(pred_scores, 2, 1)
    pos_scores = m_dev_preds[0]
    pos_scores = pos_scores[torch.arange(0, mapped_triples.shape[0]),
                            mapped_triples[:, 0]]
    neg_scores, neg_index_topk = get_neg_scores_top_k(mapped_triples, m_dev_preds, all_pos_triples, top_k) # [[h1 * candidate, h2 * candicate...][t1,t2...]]
    torch.save(pos_scores, out_dir + "eval_pos_scores.pt")
    torch.save(neg_scores, out_dir + "eval_neg_scores.pt")
    torch.save(neg_index_topk, out_dir + "eval_neg_index.pt")


def calc_hit_at_k(pred_scores, ground_truth_idx):
    """Calculates mean number of hits@k. Higher values are ranked first.
    the pos scores have been masked
    Returns: list of float, of the same length as hit_positions, containing
        Hits@K score.

    """
    # scores[torch.arange(0, batch.shape[0]), batch[:, column]]
    targets = pred_scores[torch.arange(0, pred_scores.shape[0]), ground_truth_idx].unsqueeze(1)
    ranks = torch.zeros(pred_scores.shape[0])
    ranks[:] += torch.sum((pred_scores >= targets).float(), dim=1).cpu()
    hits_at = list(map(
        lambda x: torch.mean((ranks <= x).float()).item(),
        [1,3,10]
    ))
    mrr = torch.mean(1. / ranks).item()
    hits_at.append(mrr)
    return hits_at


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
        heads = torch.as_tensor(g['h'].values)
        tails = torch.as_tensor(g['t'].values)
        head_hits = calc_hit_at_k(h_preds, heads)
        tail_hits = calc_hit_at_k(t_preds, tails)
        both_hit = calc_hit_at_k(torch.cat([h_preds, t_preds]), torch.cat((heads, tails)))
        head_tail_eval.append(torch.as_tensor([head_hits,
                              tail_hits,
                              both_hit]))
    head_tail_eval = torch.stack(head_tail_eval, 0)
    torch.save(head_tail_eval, out_dir + file_name)


def per_mapping_eval(pykeen_dataset, scores, out_dir):
    mappings = ['1-1', '1-n', 'n-1', 'n-m']
    rel_mappings = find_relation_mappings(pykeen_dataset)
    dev = pykeen_dataset.validation.mapped_triples
    triples_df = pd.DataFrame(data=dev.numpy(), columns=['h', 'r', 't'], )
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
            g_index = torch.from_numpy(tri_group.index.values)
            rg_index = torch.as_tensor(g_index)
            h_preds, t_preds = scores[rg_index].chunk(2, 1)
            heads = g[:, 0]
            tails = g[:, 2]
            head_hits = calc_hit_at_k(h_preds, heads)
            tail_hits = calc_hit_at_k(t_preds, tails)
            both_hit = calc_hit_at_k(torch.cat([h_preds, t_preds]), torch.cat((heads, tails)))
            head_tail_eval.append(torch.as_tensor([head_hits,
                                   tail_hits,
                                   both_hit]))
        else:
            head_tail_eval.append(torch.full([3, 4], 0.001))
    torch.save(torch.stack(head_tail_eval, 0), out_dir + f"rank_mapping_rel_eval.pt")
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
        '--model', choices=models, default="CP",
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
        '--out_dir', type=str, default="../../outputs/UMLS/CP/"
    )
    m_args = parser.parse_args()
    train_and_pred(m_args)
