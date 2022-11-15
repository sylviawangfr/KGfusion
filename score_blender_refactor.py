import json
import logging
import os
from typing import Optional
import pandas as pd
from pykeen.datasets import FB15k237, Nations
from pykeen.evaluation import RankBasedEvaluator
from pykeen.constants import TARGET_TO_INDEX
from pykeen.evaluation.evaluator import Evaluator
from pykeen.evaluation.evaluator import create_sparse_positive_filter_, create_dense_positive_mask_, filter_scores_
from pykeen.typing import MappedTriples, COLUMN_HEAD, COLUMN_TAIL, Target
from pykeen.utils import resolve_device
from torch import Tensor, FloatTensor
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.data import Dataset

from blender_feature_per_rel import generate_pred_input_feature, generate_training_input_feature
from context_load_and_run import load_score_context
from main import get_all_pos_triples

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ScoreBlenderLinear(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.linear = nn.Linear(in_features=in_dim, out_features=1)

    def forward(self, in_features):
        return self.linear(in_features)


def train_linear_blender(in_dim, pos_eval_and_scores, neg_eval_and_scores, params, work_dir):
    model = ScoreBlenderLinear(in_dim)
    loss_func = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    device: torch.device = resolve_device()
    logger.info(f"Using device: {device}")
    features = torch.cat([pos_eval_and_scores, neg_eval_and_scores], 0)
    labels = torch.cat([torch.ones(pos_eval_and_scores.shape[0], 1), torch.zeros(neg_eval_and_scores.shape[0], 1)], 0)
    if device is not None:
        model.to(device)
        features.to(device)
        labels.to(device)

    for e in range(params['epochs']):
        blended = model(features)
        loss = loss_func(blended, labels)
        print('Loss at epoch {} : {}'.format(e, loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(model, os.path.join(work_dir,
                                   'margin_ensemble.pth'))
    return model


def aggregate_scores(model, mapped_triples: MappedTriples, context_resource, all_pos_triples):
    h_t_features = generate_pred_input_feature(mapped_triples, context_resource)
    # Send to device
    device: torch.device = resolve_device()
    logger.info(f"Using device: {device}")
    if device is not None:
        model.to(device)
    # Ensure evaluation mode
    model.eval()
    # Send tensors to device
    h_t_features = h_t_features.to(device=device)
    ens_scores = model.linear(h_t_features)
    #  unwrap scores
    h_scores, t_scores = torch.chunk(ens_scores, 2, 0)
    # restore format that required by pykeen evaluator
    evaluator = RankBasedEvaluator()
    relation_filter = None
    relation_filter = restore_eval_format(
        batch=mapped_triples,
        scores=h_scores,
        target='head',
        evaluator=evaluator,
        all_pos_triples=all_pos_triples,
        relation_filter=relation_filter,
    )
    relation_filter = restore_eval_format(
        batch=mapped_triples,
        scores=t_scores,
        target='tail',
        evaluator=evaluator,
        all_pos_triples=all_pos_triples,
        relation_filter=relation_filter,
    )
    result = evaluator.finalize()
    return result


def restore_eval_format(
        batch: MappedTriples,
        scores: FloatTensor,
        target: Target,
        evaluator: Evaluator,
        all_pos_triples: Optional[MappedTriples],
        relation_filter: Optional[torch.BoolTensor],
) -> torch.BoolTensor:
    """
    Evaluate ranking for batch.
    :param scores:
    :param batch: shape: (batch_size, 3)
        The batch of currently evaluated triples.
    :param target:
        The prediction target.
    :param evaluator:
        The evaluator
    :param all_pos_triples:
        All positive triples (required if filtering is necessary).
    :param relation_filter:
        The relation filter. Can be re-used.
    :raises ValueError:
        if all positive triples are required (either due to filtered evaluation, or requiring dense masks).

    :return:
        The relation filter, which can be re-used for the same batch.
    """
    if evaluator.filtered or evaluator.requires_positive_mask:
        column = TARGET_TO_INDEX[target]
        if all_pos_triples is None:
            raise ValueError(
                "If filtering_necessary of positive_masks_required is True, all_pos_triples has to be "
                "provided, but is None."
            )

        # Create filter
        positive_filter, relation_filter = create_sparse_positive_filter_(
            hrt_batch=batch,
            all_pos_triples=all_pos_triples,
            relation_filter=relation_filter,
            filter_col=column,
        )
    else:
        positive_filter = relation_filter = None
    candidate_number = int(scores.shape[0] / batch.shape[0])
    scores = scores.reshape([batch.shape[0], candidate_number])
    if evaluator.filtered:
        assert positive_filter is not None
        # Select scores of true
        true_scores = scores[torch.arange(0, batch.shape[0]), batch[:, column]]
        # the rank-based evaluators needs the true scores with trailing 1-dim
        true_scores = true_scores.unsqueeze(dim=-1)
    else:
        true_scores = None

    # Create a positive mask with the size of the scores from the positive filter
    if evaluator.requires_positive_mask:
        assert positive_filter is not None
        positive_mask = create_dense_positive_mask_(zero_tensor=torch.zeros_like(scores), filter_batch=positive_filter)
    else:
        positive_mask = None

    # process scores
    evaluator.process_scores_(
        hrt_batch=batch,
        target=target,
        true_scores=true_scores,
        scores=scores,
        dense_positive_mask=positive_mask,
    )

    return relation_filter


def test_blender_refactor(dataset, work_dir, para):
    context = load_score_context(model_list, in_dir=work_dir)
    all_pos = get_all_pos_triples(dataset)
    p, n = generate_training_input_feature(d.validation.mapped_triples, context, all_pos)
    mo = train_linear_blender(p.shape[1], p, n, params=para, work_dir=work_dir)
    re = aggregate_scores(mo, d.testing.mapped_triples, context, all_pos)
    print(json.dumps(re.to_dict(), indent=2))
    json.dump(re.to_dict(), open(work_dir+"blended.json", "w"), indent=2)


if __name__ == '__main__':
    model_list = ['ComplEx', 'TuckER', 'RotatE']

    # para = dict(lr=0.1, weight_decay=5e-3, epochs=3, models=model_list)
    # d = Nations()
    # wdr = 'outputs/nations/'

    para = dict(lr=1, weight_decay=5e-3, epochs=500, models=model_list)
    wdr = 'outputs/fb237/'
    d = FB15k237()

    # per_rel_eval(model_list, dataset=d, work_dir=wdr)
    test_blender_refactor(d, work_dir=wdr, para=para)