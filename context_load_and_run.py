import json
import logging
from typing import List

import pykeen
import torch
from pykeen.datasets import Nations
from pykeen.pipeline import pipeline
from pykeen.evaluation import RankBasedEvaluator
import pandas as pd
from pykeen.pipeline.api import triple_hash
from pykeen.typing import MappedTriples
from pykeen.utils import resolve_device

import utils
from raw_score_evaluator import predict_head_tail_scores, per_rel_rank_evaluate
from utils import save2json


logger = logging.getLogger(__name__)


def test_multi_models_eval_individual(model_keywords: [], dataset: pykeen.datasets):
    context_resource = {m: {} for m in model_keywords}
    # group by relations
    mapped_triples_eval = dataset.validation.mapped_triples
    triples_df = pd.DataFrame(data=mapped_triples_eval.numpy(), columns=['h', 'r', 't'])
    groups = triples_df.groupby('r', group_keys=True, as_index=False)
    releval2idx = {key: idx for idx, key in enumerate(groups.groups.keys())}
    ordered_keys = releval2idx.keys()
    for m in model_keywords:
        single_model = pipeline(
            training=dataset.training,
            validation=dataset.validation,
            testing=dataset.testing,
            model=m,
            evaluator=RankBasedEvaluator,
            stopper='early'
        )
        rel_eval = per_rel_rank_evaluate(single_model.model, groups, ordered_keys)
        eval = predict_head_tail_scores(single_model.model, dataset.validation.mapped_triples, mode=None)
        preds = predict_head_tail_scores(single_model.model, dataset.testing.mapped_triples, mode=None)
        context_resource[m] = {'model': single_model, 'rel_eval': rel_eval, 'eval': eval, 'preds': preds}
    context_resource.update({'releval2idx': releval2idx, 'models': model_keywords})
    return context_resource


def get_additional_filter_triples(do_filter_validation, training, validation=None):
    # Build up a list of triples if we want to be in the filtered setting
    additional_filter_triples_names = dict()
    additional_filter_triples: List[MappedTriples] = [
            training.mapped_triples,
        ]
    # additional_filter_triples_names["training"] = triple_hash(training.mapped_triples)

    # Determine whether the validation triples should also be filtered while performing test evaluation
    if do_filter_validation:
        additional_filter_triples.append(validation.mapped_triples)
        # additional_filter_triples_names["validation"] = triple_hash(validation.mapped_triples)
    # evaluation_kwargs = {"additional_filter_triples": additional_filter_triples}
    return additional_filter_triples


def per_rel_eval(model_keywords: [], dataset: pykeen.datasets, work_dir=''):
    mapped_triples_eval = dataset.validation.mapped_triples
    triples_df = pd.DataFrame(data=mapped_triples_eval.numpy(), columns=['h', 'r', 't'])
    groups = triples_df.groupby('r', group_keys=True, as_index=False)
    releval2idx = {key: idx for idx, key in enumerate(groups.groups.keys())}
    ordered_keys = releval2idx.keys()
    device: torch.device = resolve_device()
    logger.info(f"Using device: {device}")
    evaluation_kwargs = {"additional_filter_triples": get_additional_filter_triples(False, dataset.training)}
    for m in model_keywords:
        m_dir = work_dir + m + "/checkpoint/trained_model.pkl"
        m_out_dir = work_dir + m + "/"
        single_model = torch.load(m_dir)
        single_model = single_model.to(device)
        per_rel_eval = per_rel_rank_evaluate(single_model, groups, ordered_keys, **evaluation_kwargs)
        eval_preds = predict_head_tail_scores(single_model, dataset.validation.mapped_triples, mode=None)
        test_preds = predict_head_tail_scores(single_model, dataset.testing.mapped_triples, mode=None)
        torch.save(eval_preds.detach().cpu(), m_out_dir + "eval.pt")
        torch.save(test_preds.detach().cpu(), m_out_dir + "preds.pt")
        torch.save(per_rel_eval.detach().cpu(), m_out_dir + "rel_eval.pt")
    save2json(releval2idx, work_dir + "releval2idx.json")


def load_score_context(model_list, in_dir):
    context_resource = {m: {} for m in model_list}
    for m in model_list:
        read_dir = in_dir + m + '/'
        rel_eval = torch.load(read_dir + "rel_eval.pt")
        eval = torch.load(read_dir + "eval.pt")
        preds = torch.load(read_dir + "preds.pt")
        context_resource[m] = {'rel_eval': rel_eval, 'eval': eval, 'preds': preds}
    releval2idx = utils.load_json(in_dir + "releval2idx.json")
    releval2idx = {int(k): releval2idx[k] for k in releval2idx}
    context_resource.update({'releval2idx': releval2idx, 'models': model_list})
    return context_resource