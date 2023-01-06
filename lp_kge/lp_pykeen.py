import argparse
import gc
from pykeen.evaluation import RankBasedEvaluator, ClassificationEvaluator
from pykeen.evaluation.evaluator import optional_context_manager, create_sparse_positive_filter_, filter_scores_
from pykeen.models import Model
from pykeen.typing import LABEL_HEAD, LABEL_TAIL, InductiveMode, MappedTriples, COLUMN_HEAD, COLUMN_TAIL
from pykeen.utils import (
    split_list_in_batches_iter,
)
from typing import Iterable, Mapping, Optional, cast
import numpy as np
from tqdm.autonotebook import tqdm
import logging
from typing import List
from pykeen.datasets import FB15k237, Nations, UMLS, get_dataset
import pykeen
import torch
import pandas as pd
from pykeen.typing import MappedTriples
from pykeen.utils import resolve_device
from utils import save2json
from pykeen.utils import prepare_filter_triples

logger = logging.getLogger(__name__)


def get_all_pos_triples(dataset: pykeen.datasets.Dataset):
    all_pos_triples = prepare_filter_triples(
        mapped_triples=dataset.testing.mapped_triples,
        additional_filter_triples=[
            dataset.training.mapped_triples,
            dataset.validation.mapped_triples,
        ],
    )
    return all_pos_triples


def predict_head_tail_scores(
        model: Model,
        mapped_triples: MappedTriples,
        batch_size: Optional[int] = None,
        slice_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        tqdm_kwargs: Optional[Mapping[str, str]] = None,
        *,
        mode: Optional[InductiveMode],
):
    """Evaluate metrics for model on mapped triples.
    :param model:
        The model to evaluate.
    :param mapped_triples:
        The triples on which to evaluate. The mapped triples should never contain inverse triples - these are created by
        the model class on the fly.
    :param only_size_probing:
        The evaluation is only performed for two batches to test the memory footprint, especially on GPUs.
    :param batch_size: >0
        A positive integer used as batch size. Generally chosen as large as possible. Defaults to 1 if None.
    :param slice_size: >0
        The divisor for the scoring function when using slicing.
    :param device:
        The device on which the evaluation shall be run. If None is given, use the model's device.
    :param tqdm_kwargs:
        Additional keyword based arguments passed to the progress bar.
    :return:
        the raw triple predicitons,
    """
    # Send to device
    if device is not None:
        model = model.to(device)
    device = model.device

    # Ensure evaluation mode
    model.predict_with_sigmoid = True
    model.eval()
    # Send tensors to device
    mapped_triples = mapped_triples.to(device=device)
    # Prepare batches
    if batch_size is None:
        # This should be a reasonable default size that works on most setups while being faster than batch_size=1
        batch_size = 32
        logger.info(f"No evaluation batch_size provided. Setting batch_size to '{batch_size}'.")
    batches = cast(Iterable[np.ndarray], split_list_in_batches_iter(input_list=mapped_triples, batch_size=batch_size))

    # Show progressbar
    num_triples = mapped_triples.shape[0]

    # Disable gradient tracking
    _tqdm_kwargs = dict(
        desc=f"Evaluating on {model.device}",
        total=num_triples,
        unit="triple",
        unit_scale=True,
        # Choosing no progress bar (use_tqdm=False) would still show the initial progress bar without disable=True
        disable=False
    )
    if tqdm_kwargs:
        _tqdm_kwargs.update(tqdm_kwargs)
    model_sample_results_h = []
    model_sample_results_t = []
    with optional_context_manager(True, tqdm(**_tqdm_kwargs)) as progress_bar, torch.inference_mode():
        # batch-wise processing
        for batch in batches:
            batch_size = batch.shape[0]
            h_scores = model.predict(hrt_batch=batch, target=LABEL_HEAD, slice_size=slice_size, mode=mode)
            model_sample_results_h.append(h_scores)
            t_scores = model.predict(hrt_batch=batch, target=LABEL_TAIL, slice_size=slice_size, mode=mode)
            model_sample_results_t.append(t_scores)
            progress_bar.update(batch_size)
    model_sample_results_h = torch.vstack(model_sample_results_h)
    model_sample_results_t = torch.vstack(model_sample_results_t)
    model_sample_results = torch.cat((model_sample_results_h, model_sample_results_t), 1)
    return model_sample_results.detach().cpu()


# def find_clusters(model: Model,
#                   mapped_triples,
#                   all_pos_triples=None,
#                   top_k=10
#                   ):
#     eval_preds = predict_head_tail_scores(model, mapped_triples,
#                                           mode=None)  # head_preds + t_preds
#     dev_predictions = torch.chunk(eval_preds, 2, 1)
#     pos_scores = dev_predictions[0]
#     pos_scores = pos_scores[torch.arange(0, mapped_triples.shape[0]),
#                             mapped_triples[:, 0]]
#     # pos_scores = torch.unsqueeze(pos_scores, 1)
#     targets = [COLUMN_HEAD, COLUMN_TAIL]
#     ht_hits = []
#     ht_fails = []
#     for index in range(2):
#         # exclude positive triples
#         positive_filter, _ = create_sparse_positive_filter_(
#             hrt_batch=mapped_triples,
#             all_pos_triples=all_pos_triples,
#             relation_filter=None,
#             filter_col=targets[index],
#         )
#         scores = filter_scores_(scores=dev_predictions[index], filter_batch=positive_filter)
#         # The scores for the true triples have to be rewritten to the scores tensor
#         scores[torch.arange(0, mapped_triples.shape[0]), mapped_triples[:, targets[index]]] = pos_scores
#         select_range = top_k if top_k < scores.shape[-1] else scores.shape[-1]
#         scores_k, indices_k = torch.nan_to_num(scores, nan=-999.).topk(k=select_range)
#         hits = (scores_k == torch.unsqueeze(pos_scores, -1))
#         hit_index = torch.unique(hits.nonzero()[:, 0])
#         triple_index = torch.arange(0, mapped_triples.shape[0])
#         combined = torch.cat((hit_index, triple_index))
#         uniques, counts = combined.unique(return_counts=True)
#         fail_index = uniques[counts == 1]
#         ht_hits.append(hit_index.detach().cpu())
#         ht_fails.append(fail_index.detach().cpu())
#     return ht_hits, ht_fails

def get_evaluator(keyword="f1"):
    clz = {
        "f1": classification_evaluate,
        "rank": rank_hits_evaluate}
    return clz[keyword]


def rank_hits_evaluate(model: Model,
                            mapped_triples,
                            batch_size: Optional[int] = None,
                            slice_size: Optional[int] = None,
                            **kwargs,
                            ):
    targets = kwargs["targets"]
    evaluator = RankBasedEvaluator()
    metrix_result_g = evaluator.evaluate(model, mapped_triples, batch_size=batch_size, slice_size=slice_size, **kwargs)
    tmp_eval = metrix_result_g.data
    if len(targets) == 2:
        return [tmp_eval[('hits_at_10', 'head', 'realistic')],
                              tmp_eval[('hits_at_10', 'tail', 'realistic')],
                              tmp_eval[('hits_at_10', 'both', 'realistic')]]
    else:
        return [tmp_eval[('hits_at_10', targets[0], 'realistic')]]


def classification_evaluate(model: Model,
                       mapped_triples,
                       batch_size: Optional[int] = None,
                       slice_size: Optional[int] = None,
                       **kwargs,
                       ):
    targets = kwargs['targets']
    result = []
    score_key = "f1_score"
    if len(targets) == 2:
        for t in targets:
            evaluator = ClassificationEvaluator()
            kwargs.update({"targets": [t]})
            metrix_result = evaluator.evaluate(model, mapped_triples, batch_size=batch_size, slice_size=slice_size, **kwargs)
            result.append(metrix_result.data[score_key])
    evaluator = ClassificationEvaluator()
    kwargs.update({"targets": [LABEL_HEAD, LABEL_TAIL]})
    metrix_result = evaluator.evaluate(model, mapped_triples, **kwargs)
    result.append(metrix_result.data[score_key])
    return result


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


def get_neg_scores_top_k(mapped_triples, dev_predictions, all_pos_triples, top_k):
    # Create filter
    targets = [LABEL_HEAD, LABEL_TAIL]
    neg_scores = []
    neg_index = []
    for index in range(2):
        # exclude positive triples
        positive_filter, _ = create_sparse_positive_filter_(
            hrt_batch=mapped_triples,
            all_pos_triples=all_pos_triples,
            relation_filter=None,
            filter_col=targets[index],
        )
        scores = filter_scores_(scores=dev_predictions[index], filter_batch=positive_filter)
        # random pick top_k negs, if no more than top_k, then fill with -999.
        # However the top_k from different model is not always the same
        # Our solution is that we pick the top_k * 2 candidates, and pick the most frequent index
        select_range = top_k if top_k < scores.shape[-1] else scores.shape[-1]
        scores_k, indices_k = torch.nan_to_num(scores, nan=-999.).topk(k=select_range)
        # remove -999 from scores_k
        nan_index = torch.nonzero(scores_k == -999.)
        indices_k[nan_index[:, 0], nan_index[:, 1]] = int(-1)
        neg_scores.append(scores_k)
        neg_index.append(indices_k)
    neg_scores = torch.stack(neg_scores, 1)  # [h1* candi,h2 * candi...,t1 * candi, t2* candi...]
    neg_index = torch.stack(neg_index, 1)
    return neg_scores, neg_index


def find_relation_mappings(dataset: pykeen.datasets.Dataset):
    all_triples = torch.cat([dataset.training.mapped_triples,
                             dataset.validation.mapped_triples,
                             dataset.testing.mapped_triples], 0)
    df = pd.DataFrame(data=all_triples.numpy(), columns=['h', 'r', 't'])
    del all_triples
    hr = df[['h', 'r']]
    possible_one_to_many_rel = hr[hr.duplicated()][['r']].drop_duplicates(keep='first')
    tmp_r1 = hr[['r']].drop_duplicates(keep='first')
    possible_one2one_1 = pd.concat([tmp_r1, possible_one_to_many_rel]).drop_duplicates(keep=False)
    rt = df[['r', 't']]
    del df
    possible_many_to_one_rel = rt[rt.duplicated()][['r']].drop_duplicates(keep='first')
    tmp_r2 = rt[['r']].drop_duplicates(keep='first')
    possible_one2one_2 = pd.concat([tmp_r2, possible_many_to_one_rel]).drop_duplicates(keep=False)
    many_to_many_rel = pd.concat([possible_one_to_many_rel, possible_many_to_one_rel])
    many_to_many_rel = many_to_many_rel[many_to_many_rel.duplicated()]
    one_to_one_rel = pd.concat([possible_one2one_1, possible_one2one_2])
    one_to_one_rel = one_to_one_rel[one_to_one_rel.duplicated()]
    one_to_many_rel = pd.concat([possible_one_to_many_rel, many_to_many_rel]).drop_duplicates(keep=False)
    many_to_one_rel = pd.concat([possible_many_to_one_rel, many_to_many_rel]).drop_duplicates(keep=False)
    all_count = len(one_to_one_rel.index) + \
                len(one_to_many_rel.index) + \
                len(many_to_one_rel.index) + \
                len(many_to_many_rel.index)
    assert(all_count == dataset.num_relations)
    rel_groups = {'one_to_one': one_to_one_rel['r'].values,
                  'one_to_many': one_to_many_rel['r'].values,
                  'many_to_one': many_to_one_rel['r'].values,
                  'many_to_many': many_to_many_rel['r'].values}
    del hr
    del rt
    gc.collect()
    return rel_groups


class LpKGE:
    def __init__(self, dataset, models, work_dir):
        self.dataset = get_dataset(dataset=dataset)
        self.models = models
        self.work_dir = work_dir

    # def find_cluster(self):
    #     mapped_triples_eval = self.dataset.validation.mapped_triples
    #     device: torch.device = resolve_device()
    #     logger.info(f"Using device: {device}")
    #     all_pos_triples = get_all_pos_triples(self.dataset)
    #     for m in self.models:
    #         m_dir = self.work_dir + m + "/checkpoint/trained_model.pkl"
    #         m_out_dir = self.work_dir + m + "/"
    #         single_model = torch.load(m_dir)
    #         single_model = single_model.to(device)
    #         ht_hits_index, ht_fails_index = find_clusters(single_model, mapped_triples_eval, all_pos_triples=all_pos_triples)
    #         torch.save(ht_hits_index[0], m_out_dir + "h_hits_index.pt")
    #         torch.save(ht_fails_index[0], m_out_dir + "h_fails_index.pt")
    #         torch.save(ht_hits_index[1], m_out_dir + "t_hits_index.pt")
    #         torch.save(ht_fails_index[1], m_out_dir + "t_fails_index.pt")

    def dev_rel_eval(self, evaluator_key):
        mapped_triples_eval = self.dataset.validation.mapped_triples
        triples_df = pd.DataFrame(data=mapped_triples_eval.numpy(), columns=['h', 'r', 't'])
        groups = triples_df.groupby('r', group_keys=True, as_index=False)
        releval2idx = {key: idx for idx, key in enumerate(groups.groups.keys())}
        ordered_keys = releval2idx.keys()
        device: torch.device = resolve_device()
        logger.info(f"Using device: {device}")
        evaluation_kwargs = {"additional_filter_triples": get_additional_filter_triples(False, self.dataset.training),
                             "targets": [LABEL_HEAD, LABEL_TAIL]}
        for m in self.models:
            m_dir = self.work_dir + m + "/checkpoint/trained_model.pkl"
            m_out_dir = self.work_dir + m + "/"
            single_model = torch.load(m_dir)
            single_model = single_model.to(device)
            rel_evals = []
            for rel in ordered_keys:
                g = groups.get_group(rel)
                g_tensor = torch.from_numpy(g.values)
                evaluator_fun = get_evaluator(evaluator_key)
                head_tail_eval = evaluator_fun(single_model, g_tensor, **evaluation_kwargs)
                rel_evals.append(head_tail_eval)
            torch.save(torch.as_tensor(rel_evals), m_out_dir + "rel_eval.pt")
        save2json(releval2idx, self.work_dir + "releval2idx.json")

    def dev_ent_eval(self, evaluator_key='f1'):
        mapped_triples_eval = self.dataset.validation.mapped_triples
        triples_df = pd.DataFrame(data=mapped_triples_eval.numpy(), columns=['h', 'r', 't'])
        h_groups = triples_df.groupby('h', group_keys=True, as_index=False)
        h_ent2idx = {key: idx for idx, key in enumerate(h_groups.groups.keys())}
        t_groups = triples_df.groupby('t', group_keys=True, as_index=False)
        t_ent2idx = {key: idx for idx, key in enumerate(t_groups.groups.keys())}
        h_ent_keys = h_ent2idx.keys()
        t_ent_keys = t_ent2idx.keys()
        device: torch.device = resolve_device()
        logger.info(f"Using device: {device}")
        evaluation_kwargs = {"additional_filter_triples": get_additional_filter_triples(False, self.dataset.training)}
        evaluator_fun = get_evaluator(evaluator_key)
        for m in self.models:
            m_dir = self.work_dir + m + "/checkpoint/trained_model.pkl"
            m_out_dir = self.work_dir + m + "/"
            single_model = torch.load(m_dir)
            single_model = single_model.to(device)
            h_ent_eval = []
            t_ent_eval = []
            for key in h_ent_keys:
                g = h_groups.get_group(key)
                g_tensor = torch.from_numpy(g.values)
                evaluation_kwargs.update({"targets": [LABEL_TAIL]})
                tmp_h_eval = evaluator_fun(single_model, g_tensor, **evaluation_kwargs)
                h_ent_eval.append(tmp_h_eval)
            for key in t_ent_keys:
                g = t_groups.get_group(key)
                g_tensor = torch.from_numpy(g.values)
                evaluation_kwargs.update({"targets": [LABEL_HEAD]})
                tmp_t_eval = evaluator_fun(single_model, g_tensor, **evaluation_kwargs)
                t_ent_eval.append(tmp_t_eval)
            h_ent_eval = torch.as_tensor(h_ent_eval)
            t_ent_eval = torch.as_tensor(t_ent_eval)
            torch.save(h_ent_eval, m_out_dir + "h_ent_eval.pt")
            torch.save(t_ent_eval, m_out_dir + "t_ent_eval.pt")
        save2json(h_ent2idx, self.work_dir + "h_ent2idx.json")
        save2json(t_ent2idx, self.work_dir + "t_ent2idx.json")

    def dev_eval(self, evaluator_key):
        mapped_triples_eval = self.dataset.validation.mapped_triples
        device: torch.device = resolve_device()
        logger.info(f"Using device: {device}")
        evaluation_kwargs = {"additional_filter_triples": get_additional_filter_triples(False, self.dataset.training),
                             "targets": [LABEL_HEAD, LABEL_TAIL]}
        result = []
        evaluator_fun = get_evaluator(evaluator_key)
        for m in self.models:
            m_dir = self.work_dir + m + "/checkpoint/trained_model.pkl"
            single_model = torch.load(m_dir)
            single_model = single_model.to(device)
            result.append(evaluator_fun(single_model, mapped_triples_eval, **evaluation_kwargs))
            print(result)

    def dev_mapping_eval(self, evaluator_key):
        mappings = ['one_to_one', 'one_to_many', 'many_to_one', 'many_to_many']
        rel_mappings = find_relation_mappings(self.dataset)
        dev = self.dataset.validation.mapped_triples
        triples_df = pd.DataFrame(data=dev.numpy(), columns=['h', 'r', 't'])
        relmapping2idx = dict()
        for idx, mapping_type in enumerate(mappings):
            mapped_rels = rel_mappings[mapping_type]
            relmapping2idx.update({int(rel): idx for rel in mapped_rels})
        device: torch.device = resolve_device()
        logger.info(f"Using device: {device}")
        evaluation_kwargs = {"additional_filter_triples": get_additional_filter_triples(False, self.dataset.training),
                             "targets": [LABEL_HEAD, LABEL_TAIL]}
        evaluator_fun = get_evaluator(evaluator_key)
        for m in self.models:
            m_dir = self.work_dir + m + "/checkpoint/trained_model.pkl"
            m_out_dir = self.work_dir + m + "/"
            single_model = torch.load(m_dir)
            single_model = single_model.to(device)
            model_eval = []
            for rel_group in mappings:
                tmp_rels = rel_mappings[rel_group]
                tri_group = triples_df.query('r in @tmp_rels')
                if len(tri_group.index) > 0:
                    mapped_group = torch.from_numpy(tri_group.values)
                    group_eval = evaluator_fun(single_model, mapped_group, **evaluation_kwargs)
                    model_eval.append(group_eval)
                else:
                    model_eval.append([0.01, 0.01, 0.01])
            torch.save(torch.Tensor(model_eval), m_out_dir + "mapping_eval.pt")
        save2json(relmapping2idx, self.work_dir + "relmapping2idx.json")

    def dev_pred(self, top_k):
        device: torch.device = resolve_device()
        logger.info(f"Using device: {device}")
        all_pos_triples = get_all_pos_triples(self.dataset)
        for m in self.models:
            m_dir = self.work_dir + m + "/checkpoint/trained_model.pkl"
            m_out_dir = self.work_dir + m + "/"
            single_model = torch.load(m_dir)
            single_model = single_model.to(device)
            # pykeen KGE dev prediction scores, ordered by scores. we need both pos scores and neg scores, and there index
            eval_preds = predict_head_tail_scores(single_model, self.dataset.validation.mapped_triples,
                                                  mode=None)  # head_preds + t_preds
            m_dev_preds = torch.chunk(eval_preds, 2, 1)
            pos_scores = m_dev_preds[0]
            pos_scores = pos_scores[torch.arange(0, self.dataset.validation.mapped_triples.shape[0]),
                                    self.dataset.validation.mapped_triples[:, 0]]
            neg_scores, neg_index_topk = get_neg_scores_top_k(self.dataset.validation.mapped_triples, m_dev_preds,
                                                              all_pos_triples,
                                                              top_k)  # [[h1 * candidate, h2 * candicate...][t1,t2...]]
            torch.save(pos_scores, m_out_dir + "eval_pos_scores.pt")
            torch.save(neg_scores, m_out_dir + "eval_neg_scores.pt")
            torch.save(neg_index_topk, m_out_dir + "eval_neg_index.pt")
            # save all scores for testing set
            test_preds = predict_head_tail_scores(single_model, self.dataset.testing.mapped_triples, mode=None)
            torch.save(test_preds, m_out_dir + "preds.pt")

    def test_pred(self):
        device: torch.device = resolve_device()
        logger.info(f"Using device: {device}")
        for m in self.models:
            m_dir = self.work_dir + m + "/checkpoint/trained_model.pkl"
            m_out_dir = self.work_dir + m + "/"
            single_model = torch.load(m_dir)
            single_model = single_model.to(device)
            # save all scores for testing set
            test_preds = predict_head_tail_scores(single_model, self.dataset.testing.mapped_triples, mode=None)
            torch.save(test_preds, m_out_dir + "preds.pt")

    # def analyze_clusters(self):
    #     context_resource = {m: {} for m in self.models}
    #     for m in self.models:
    #         read_dir = self.work_dir + m + '/'
    #         h_hits_index = torch.load(read_dir + "h_hits_index.pt")
    #         t_hits_index = torch.load(read_dir + "t_hits_index.pt")
    #         h_fails_index = torch.load(read_dir + "h_fails_index.pt")
    #         t_fails_index = torch.load(read_dir + "t_fails_index.pt")
    #         h_hits_ent = self.dataset.validation.mapped_triples[h_hits_index, 2]
    #         h_hits_rel = self.dataset.validation.mapped_triples[h_hits_index, 1]
    #         h_fails_ent = self.dataset.validation.mapped_triples[h_fails_index, 2]
    #         h_fails_rel = self.dataset.validation.mapped_triples[h_fails_index, 1]
    #         t_hits_ent = self.dataset.validation.mapped_triples[t_hits_index, 0]
    #         t_hits_rel = self.dataset.validation.mapped_triples[t_hits_index, 1],
    #         h_ent_uniques, h_ent_counts = h_hits_ent.unique(return_counts=True)
    #         h_ent_fail_uniques, h_ent_fail_counts = h_fails_ent.unique(return_counts=True)
    #         h_rel_uniques, h_rel_counts = h_hits_rel.unique(return_counts=True)
    #         h_rel_fail_uniques, h_rel_fail_counts = h_fails_rel.unique(return_counts=True)
    #         # context_resource[m] = {'h_hits': self.dataset.validation.mapped_triples[h_hits, 1:],
    #         #                        't_hits': self.dataset.validation.mapped_triples[t_hits, 0:2]}
    #
    #     # get r,t sets
    #     all_h_hits = torch.cat([v['h_hits'] for k, v in context_resource.items()], 0)
    #     h_hits_uniques, h_hits_counts = all_h_hits.unique(return_counts=True)
    #
    #     all_t_hits = torch.cat([v['t_hits'] for k,v in context_resource.items()], 0)
    #     all_h_fails = torch.cat([v['h_fails'] for k,v in context_resource.items()], 0)
    #     all_t_fails = torch.cat([v['t_fails'] for k,v in context_resource.items()], 0)
    #     h_hits_uniques, h_hits_counts = all_h_hits.unique(return_counts=True)
    #     t_hits_uniques, t_hits_counts = all_t_hits.unique(return_counts=True)
    #     h_fails_uniques, h_fails_counts = all_h_fails.unique(return_counts=True)
    #     t_fails_uniques, t_fails_counts = all_t_fails.unique(return_counts=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="ComplEx_TuckER")
    # parser.add_argument('--models', type=str, default="NodePiece")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    parser.add_argument('--evaluator_key', type=str, default="f1")
    args = parser.parse_args()
    param1 = args.__dict__
    param1.update({"models": args.models.split('_')})
    pykeen_lp = LpKGE(dataset=param1['dataset'], models=param1['models'], work_dir=param1['work_dir'])
    eval_key = param1['evaluator_key']
    pykeen_lp.dev_eval(eval_key)
    # pykeen_lp.dev_rel_eval(eval_key)
    # pykeen_lp.dev_ent_eval(eval_key)
    # pykeen_lp.dev_mapping_eval(eval_key)
    # pykeen_lp.dev_pred(top_k=100)
    # pykeen_lp.test_pred()
    # find_relation_mappings(d)
