from pykeen.evaluation import RankBasedEvaluator
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
from pykeen.datasets import FB15k237, Nations, UMLS
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


def per_rel_rank_evaluate(model: Model,
                          mapped_triples_groups,
                          ordered_keys,
                          batch_size: Optional[int] = None,
                          slice_size: Optional[int] = None,
                          **kwargs,
                          ):
    evaluator = RankBasedEvaluator()
    per_group_eval = dict()
    for key in ordered_keys:
        g = mapped_triples_groups.get_group(key)
        g_tensor = torch.from_numpy(g.values)
        metrix_result_g = evaluator.evaluate(model, g_tensor, batch_size=batch_size, slice_size=slice_size, **kwargs)
        per_group_eval.update({key: metrix_result_g})
    head_tail_mrr = []
    for key in ordered_keys:
        tmp_eval = per_group_eval[key].data
        head_tail_mrr.append([tmp_eval[('hits_at_10', 'head', 'realistic')],
                              tmp_eval[('hits_at_10', 'tail', 'realistic')],
                              tmp_eval[('hits_at_10', 'both', 'realistic')]])
    head_tail_mrr = torch.Tensor(head_tail_mrr)
    return head_tail_mrr


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
    targets = [COLUMN_HEAD, COLUMN_TAIL]
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
    neg_scores = torch.stack(neg_scores, 1) # [h1* candi,h2 * candi...,t1 * candi, t2* candi...]
    neg_index = torch.stack(neg_index, 1)
    return neg_scores, neg_index


class LpKGE:
    def __init__(self, dataset, models, work_dir):
        self.dataset = dataset
        self.models = models
        self.work_dir = work_dir

    def dev_eval(self):
        mapped_triples_eval = self.dataset.validation.mapped_triples
        triples_df = pd.DataFrame(data=mapped_triples_eval.numpy(), columns=['h', 'r', 't'])
        groups = triples_df.groupby('r', group_keys=True, as_index=False)
        releval2idx = {key: idx for idx, key in enumerate(groups.groups.keys())}
        ordered_keys = releval2idx.keys()
        device: torch.device = resolve_device()
        logger.info(f"Using device: {device}")
        evaluation_kwargs = {"additional_filter_triples": get_additional_filter_triples(False, self.dataset.training)}
        for m in self.models:
            m_dir = self.work_dir + m + "/checkpoint/trained_model.pkl"
            m_out_dir = self.work_dir + m + "/"
            single_model = torch.load(m_dir)
            single_model = single_model.to(device)
            per_rel_eval = per_rel_rank_evaluate(single_model, groups, ordered_keys, **evaluation_kwargs)
            torch.save(per_rel_eval.detach().cpu(), m_out_dir + "rel_eval.pt")
        save2json(releval2idx, self.work_dir + "releval2idx.json")

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
            eval_preds = predict_head_tail_scores(single_model, self.dataset.validation.mapped_triples, mode=None) # head_preds + t_preds
            m_dev_preds = torch.chunk(eval_preds, 2, 1)
            pos_scores = m_dev_preds[0]
            pos_scores = pos_scores[torch.arange(0, self.dataset.validation.mapped_triples.shape[0]),
                                    self.dataset.validation.mapped_triples[:, 0]]
            neg_scores, neg_index_topk = get_neg_scores_top_k(self.dataset.validation.mapped_triples, m_dev_preds, all_pos_triples, top_k) # [[h1 * candidate, h2 * candicate...][t1,t2...]]
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


if __name__ == '__main__':
    # d = Nations()
    # dirtmp = 'outputs/nations/'
    # d = FB15k237()
    # dirtmp = 'outputs/fb237/'
    d = UMLS()
    dirtmp = '../outputs/umls/'
    pykeen_lp = LpKGE(dataset=d, models=['ComplEx', 'TuckER'], work_dir=dirtmp)
    pykeen_lp.dev_pred(top_k=100)
    pykeen_lp.dev_eval()
    pykeen_lp.test_pred()
