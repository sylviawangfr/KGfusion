import logging

import torch
from netcal.scaling import LogisticCalibration
from pykeen.datasets import Nations
from pykeen.evaluation import RankBasedEvaluator
from pykeen.models import Model
from pykeen.models.uncertainty import predict_hrt_uncertain
from pykeen.pipeline import pipeline
from pykeen.typing import LABEL_HEAD, LABEL_TAIL
from pykeen.utils import resolve_device

from features.FusionDataset import FusionDataset
from context_load_and_run import load_score_context
from features.feature_scores_only_dataset import ScoresOnlyDataset
from features.feature_scores_with_index_dataset import TopKIndexDataset
from main import get_all_pos_triples
from score_blender_refactor import restore_eval_format
from common_utils import format_result


logger = logging.getLogger(__name__)


def platt_scaling_all_models(train_feature_dataset: FusionDataset, pred_feature_dataset:FusionDataset, all_pos_triples):
    logistic = LogisticCalibration(method='mle', detection=True, )
    # detection : bool, default: False
    #     If True, the input array 'X' is treated as a box predictions with several box features (at least
    # box confidence must be present) with shape (n_samples, [n_box_features]).
    pos, neg = train_feature_dataset.get_all_dev_examples()
    input = torch.cat([pos, neg], 0).numpy()
    labels = torch.cat([torch.ones(pos.shape[0], 1),
                        torch.zeros(neg.shape[0], 1)], 0).numpy()
    logistic.fit(input, labels)
    pred_features = pred_feature_dataset.get_all_test_examples()
    ens_logits = logistic.transform(pred_features)
    h_preds, t_preds = torch.chunk(torch.as_tensor(ens_logits), 2, 0)
    # restore format that required by pykeen evaluator
    ht_scores = [h_preds, t_preds]
    evaluator = RankBasedEvaluator()
    relation_filter = None
    for ind, target in enumerate([LABEL_HEAD, LABEL_TAIL]):
        relation_filter = restore_eval_format(
            batch=pred_feature_dataset.mapped_triples,
            scores=ht_scores[ind],
            target=target,
            evaluator=evaluator,
            all_pos_triples=all_pos_triples,
            relation_filter=relation_filter,
        )
    result = evaluator.finalize()
    print(format_result(result))
    return result


def test_uncertainty():
    result = pipeline(dataset="nations", model="ERMLPE", loss="bcewithlogits")
    prediction_with_uncertainty = predict_hrt_uncertain(
        model=result.model,
        hrt_batch=result.training.mapped_triples,
        num_samples=100,
    )
    df = result.training.tensor_to_df(
        logits=prediction_with_uncertainty.score[:, 0],
        probability=prediction_with_uncertainty.score[:, 0].sigmoid(),
        uncertainty=prediction_with_uncertainty.uncertainty[:, 0],
    )
    print(df.nlargest(5, columns="uncertainty"))


def get_scores_probs_uncertainty(model: Model, mapped_triples):
    prediction_with_uncertainty = predict_hrt_uncertain(model=model,
                                                        hrt_batch=mapped_triples)
    logits=prediction_with_uncertainty.score[:, 0]
    probability=prediction_with_uncertainty.score[:, 0].sigmoid()
    uncertainty=prediction_with_uncertainty.uncertainty[:, 0]
    re = torch.stack([logits, probability, uncertainty], 0).detach().cup()
    return re


def get_scores_probs_uncertenty_for_dev_samples(dataset, paras):
    # 1. load individual model context
    work_dir = paras['work_dir']
    context_resources = load_score_context(paras['models'], in_dir=work_dir)
    device: torch.device = resolve_device()
    logger.info(f"Using device: {device}")
    # get top_K most close neg samples
    all_pos = get_all_pos_triples(dataset)
    dev_dataset = TopKIndexDataset(dataset.validation.mapped_triples, context_resources, all_pos,
                             num_neg=paras['num_neg'])
    in_features = dev_dataset.get_pos_and_top_k_neg()
    all_spu = []
    for m in enumerate(paras['models']):
        m_dir = work_dir + m + "/checkpoint/trained_model.pkl"
        m_out_dir = work_dir + m + "/"
        single_model = torch.load(m_dir)
        single_model = single_model.to(device)
        re_m = get_scores_probs_uncertainty(single_model, in_features)
        all_spu.append(dev_dataset.split_features(re_m))
    return all_spu





if __name__ == '__main__':
    d = Nations()
    dirtmp = 'outputs/nations/'
    # predict_train_dev(['ComplEx', 'TuckER', 'RotatE'], d, work_dir='outputs/nations/')
    # d = FB15k237()
    context = load_score_context(['ComplEx', 'TuckER', 'RotatE'], in_dir=dirtmp)
    all_pos = get_all_pos_triples(d)
    d2 = ScoresOnlyDataset(d.validation.mapped_triples, context, all_pos, num_neg=4)
    d3 = ScoresOnlyDataset(d.testing.mapped_triples, context, all_pos)
    platt_scaling_all_models(d2, d3, all_pos)