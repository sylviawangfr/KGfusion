import argparse
from pykeen.evaluation import RankBasedEvaluator
from pykeen.evaluation.evaluator import optional_context_manager
from pykeen.models import Model
from pykeen.typing import LABEL_HEAD, LABEL_TAIL, InductiveMode
from pykeen.utils import (
    split_list_in_batches_iter,
)
from typing import Iterable, Mapping, Optional, cast
import numpy as np
from tqdm.autonotebook import tqdm
import logging
from typing import List
from pykeen.datasets import get_dataset
import torch
from pykeen.typing import MappedTriples
from pykeen.utils import resolve_device

from analysis.group_eval_utils import to_fusion_eval_format, get_all_pos_triples


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


# def get_evaluator_cls(keyword="f1"):
#     clz = {
#         "f1": GroupededClassificationEvaluator,
#         "rank": GroupedRankBasedEvaluator}
#     return clz[keyword]


class LpKGE:
    def __init__(self, dataset, models, work_dir):
        self.dataset = get_dataset(dataset=dataset)
        self.models = models
        self.work_dir = work_dir

    def pred(self):
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
            to_fusion_eval_format(self.dataset, eval_preds, all_pos_triples, m_out_dir, 100)
            # save all scores for testing set
            test_preds = predict_head_tail_scores(single_model, self.dataset.testing.mapped_triples, mode=None)
            torch.save(test_preds, m_out_dir + "preds.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="TuckER_RotatE")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    args = parser.parse_args()
    param1 = args.__dict__
    param1.update({"models": args.models.split('_')})
    pykeen_lp = LpKGE(dataset=param1['dataset'], models=param1['models'], work_dir=param1['work_dir'])
    pykeen_lp.pred()
