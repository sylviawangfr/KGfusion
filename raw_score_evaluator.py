from pykeen.evaluation import RankBasedEvaluator
from pykeen.evaluation.evaluator import optional_context_manager
from pykeen.models import Model
from pykeen.typing import LABEL_HEAD, LABEL_TAIL, InductiveMode, MappedTriples
from pykeen.utils import (
    split_list_in_batches_iter,
)
import logging
from typing import Iterable, Mapping, Optional, cast
import numpy as np
import torch
from tqdm.autonotebook import tqdm


logger = logging.getLogger(__name__)


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
    return model_sample_results


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
        head_tail_mrr.append([tmp_eval[('adjusted_hits_at_k', 'head', 'realistic')],
                              tmp_eval[('adjusted_hits_at_k', 'tail', 'realistic')]])
    head_tail_mrr = torch.Tensor(head_tail_mrr)
    return head_tail_mrr
