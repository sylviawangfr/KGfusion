import argparse
import json
import logging
import os
from typing import Optional
import gc
import torch.nn.functional as F
from mlflow.entities import ViewType
from pykeen.datasets import Dataset, get_dataset
from pykeen.evaluation import RankBasedEvaluator
from pykeen.constants import TARGET_TO_INDEX
from pykeen.evaluation.evaluator import Evaluator
from pykeen.evaluation.evaluator import create_sparse_positive_filter_, create_dense_positive_mask_
from pykeen.typing import MappedTriples, Target, LABEL_HEAD, LABEL_TAIL
from pykeen.utils import resolve_device
from torch import FloatTensor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange
from mlflow import log_param
import mlflow.pytorch
from common_utils import format_result
from context_load_and_run import load_score_context
from features.feature_per_rel_ht1_dataset import PerRelSignalDataset
from features.feature_per_rel_ht2_dataset import PerRelNoSignalDataset
from features.feature_scores_only_dataset import ScoresOnlyDataset
from main import get_all_pos_triples

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScoreBlenderLinear1(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.linear1 = nn.Linear(in_features=in_dim, out_features=1)

    def forward(self, in_features):
        return self.linear1(in_features)


class ScoreBlenderLinear2(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.linear1 = nn.Linear(in_features=in_dim, out_features=in_dim * 2)
        self.linear2 = nn.Linear(in_features=in_dim * 2, out_features=1)

    def forward(self, in_features):
        return self.linear2(self.linear1(in_features))


class ScoreBlenderLinear3(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.linear1 = nn.Linear(in_features=in_dim, out_features=in_dim * 2)
        self.linear2 = nn.Linear(in_features=in_dim * 2, out_features=1)
        self.activate = nn.ReLU()

    def forward(self, in_features):
        return self.linear2(self.activate(self.linear1(in_features)))


def train_step_linear_blender_balanced_BCE(model, dataloader:DataLoader, device, params):
    num_pos = dataloader.dataset.num_triples
    num_neg = num_pos * dataloader.dataset.num_neg
    alpha_pos = num_neg / (num_pos + num_neg)
    alpha_neg = num_pos * 1.1 / (num_neg + num_pos)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=params['lr'], momentum=params['momentum'])
    train_loss = 0
    for batch, (pos_feature, neg_feature) in enumerate(dataloader):
        features = torch.cat([pos_feature, neg_feature], 0)
        features = features.to(device)
        labels = torch.cat([torch.ones(pos_feature.shape[0], 1), torch.zeros(neg_feature.shape[0], 1)], 0)
        labels = labels.to(device)
        y_logits = model(features)
        nn.BCEWithLogitsLoss()
        pos = torch.eq(labels, 1).float()
        neg = torch.eq(labels, 0).float()
        weights = alpha_pos * pos + alpha_neg * neg
        weights = weights.to(device)
        loss = F.binary_cross_entropy_with_logits(y_logits, labels, weights, reduction='sum')
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    gc.collect()
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    train_loss = train_loss / len(dataloader)
    return train_loss


def train_step_linear_blender_mean_BCE(model, dataloader:DataLoader, device, params):
    loss_func = nn.BCEWithLogitsLoss(reduction='sum')
    # optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    optimizer = torch.optim.SGD(params=model.parameters(), lr=params['lr'], momentum=params['momentum'])
    train_loss = 0
    for batch, (pos_feature, neg_feature) in enumerate(dataloader):
        # features = torch.cat([pos_feature, neg_feature], 0)
        # labels = torch.cat([torch.ones(pos_feature.shape[0], 1), torch.zeros(neg_feature.shape[0], 1)], 0)
        # features = features.to(device)
        # y_logits = model(features)
        # loss = loss_func(y_logits, labels)
        pos_feature = pos_feature.to(device)
        neg_feature = neg_feature.to(device)
        pos_y_logits = model(pos_feature).squeeze()
        neg_y_logits = model(neg_feature).view(len(pos_y_logits), -1, ).mean(dim=1)
        y_logits = torch.cat([pos_y_logits, neg_y_logits]).unsqueeze(1)
        labels = torch.cat([torch.ones(pos_feature.shape[0], 1), torch.zeros(pos_feature.shape[0], 1)], 0)
        labels = labels.to(device)
        loss = loss_func(y_logits, labels)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    gc.collect()
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    train_loss = train_loss / len(dataloader)
    return train_loss


def train_step_linear_blender_Margin(model, dataloader:DataLoader, device, params):
    loss_func = nn.MarginRankingLoss(margin=params['margin'], reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=params['lr'], momentum=params['momentum'], weight_decay=params['weight_decay'])
    optimizer.zero_grad()
    train_loss = 0
    for batch, (pos_feature, neg_feature) in enumerate(dataloader):
        pos_feature = pos_feature.to(device)
        neg_feature = neg_feature.to(device)
        pos = model(pos_feature)
        neg = model(neg_feature)
        loss = loss_func(pos.squeeze(), neg.view(len(pos), -1, ).mean(dim=1), torch.Tensor([1]).to(device))
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    del pos_feature
    del neg_feature
    gc.collect()
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    train_loss = train_loss / len(dataloader)
    return train_loss


def get_train_loop(loss='bce'):
    func_map = {"margin": train_step_linear_blender_Margin,
                "bce": train_step_linear_blender_mean_BCE,
                "bbce": train_step_linear_blender_balanced_BCE}
    return func_map[loss]


def get_nn_dataset(keyword='2'):
    clz = {"1": PerRelSignalDataset,
           "2": PerRelNoSignalDataset,
           "3": ScoresOnlyDataset}
    return clz[keyword]


def get_MLP(keyword='2'):
    clz = {"1": ScoreBlenderLinear1,
           "2": ScoreBlenderLinear2,
           "3": ScoreBlenderLinear3}
    return clz[keyword]

def train_aggregation_model(mapped_triples: MappedTriples, context_resource, all_pos_triples, para):
    dataset_cls = get_nn_dataset(para['dataloader'])
    train_data = dataset_cls(mapped_triples, context_resource, all_pos_triples,
                                                  num_neg=para['num_neg'])
    train_dataloader = DataLoader(train_data, batch_size=para['batch_size'], shuffle=True, collate_fn=train_data.collate_train)
    train_loop = get_train_loop(para['loss'])
    num_epochs = para['epochs']
    epochs = trange(num_epochs)
    model_cls = get_MLP(para['linear'])
    model = model_cls(train_data.dim)
    device: torch.device = resolve_device()
    # logger.info(f"Using device: {device}")
    print(f"Train Using device: {device}")
    if device is not None:
        # logger.info(f"Send model to device: {device}")
        print(f"Send model to device: {device}")
        model.to(device)
    model.train()
    for e in epochs:
        train_loss = train_loop(model, train_dataloader, device, para)
        # logger.info("loss: {}".format(train_loss))
        print("loss: {}".format(train_loss))
        epochs.set_postfix_str({"loss: {}".format(train_loss)})
        if para['mlflow']:
            mlflow.log_metric("train_loss", train_loss, step=e)
    work_dir = para['work_dir']
    torch.save(model, os.path.join(work_dir,
                                   'margin_ensemble.pth'))
    return model


def aggregate_scores(model, mapped_triples: MappedTriples, context_resource, all_pos_triples, para):
    # Send to device
    device: torch.device = resolve_device()
    logger.info(f"Eval Using device: {device}")
    if device is not None and next(model.parameters()).device != device:
        model.to(device)
    # Ensure evaluation mode
    model.eval()
    # Send tensors to device
    h_preds = []
    t_preds = []
    dataloader_cls = get_nn_dataset(para['dataloader'])
    test_per_rel_dataset = dataloader_cls(mapped_triples, context_resource, all_pos_triples)
    test_dataloader = DataLoader(test_per_rel_dataset, batch_size=32, collate_fn=test_per_rel_dataset.collate_test)
    for i, batch in enumerate(test_dataloader):
        ens_logits = model(batch.to(device))
        ht_preds = torch.chunk(ens_logits.cpu(), 2, 0)
        h_preds.append(ht_preds[0])
        t_preds.append(ht_preds[1])
    # restore format that required by pykeen evaluator
    h_preds = torch.cat(h_preds, 0)
    t_preds = torch.cat(t_preds, 0)
    ht_scores = [h_preds, t_preds]
    evaluator = RankBasedEvaluator()
    relation_filter = None
    for ind, target in enumerate([LABEL_HEAD, LABEL_TAIL]):
        relation_filter = restore_eval_format(
            batch=mapped_triples,
            scores=ht_scores[ind],
            target=target,
            evaluator=evaluator,
            all_pos_triples=all_pos_triples,
            relation_filter=relation_filter,
        )
    result = evaluator.finalize()
    if para['mlflow']:
        mlflow.log_param('result', format_result(result))
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


def pipeline_aggregation(dataset, para):
    if para['mlflow']:
        exp = mlflow.search_experiments(view_type=ViewType.ALL, filter_string="name='KGFusion'")
        if not exp:
            mlflow.create_experiment("KGFusion")
        mlflow.set_experiment("KGFusion")
        for k in para:
            log_param(k, para[k])
    # 1. load individual model context
    work_dir = para['work_dir']
    context = load_score_context(para['models'], in_dir=work_dir)
    # 2. train model
    all_pos = get_all_pos_triples(dataset)
    mo = train_aggregation_model(dataset.validation.mapped_triples, context, all_pos, para)
    # 3. aggregate scores for testing
    re = aggregate_scores(mo, dataset.testing.mapped_triples, context, all_pos, para)
    print(json.dumps(re.to_dict(), indent=2))
    json.dump(re.to_dict(), open(work_dir + "blended.json", "w"), indent=2)


if __name__ == '__main__':
    model_l = ['ComplEx', 'TuckER', 'RotatE']
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="ComplEx_TuckER_RotatE")
    parser.add_argument('--dataset', type=str, default="Nations")
    parser.add_argument('--dataloader', type=str, default="3")
    parser.add_argument('--linear', type=str, default="1")
    parser.add_argument('--work_dir', type=str, default="outputs/nations/")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--loss", type=str, default='margin')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_neg", type=int, default=4)
    parser.add_argument("--mlflow", type=str, default='True')
    parser.add_argument("--momentum", type=float, default=0.8)
    parser.add_argument("--margin", type=float, default=5)
    args = parser.parse_args()
    if args.mlflow == 'True':
        args.mlflow = True
    para = args.__dict__
    d: Dataset = get_dataset(
        dataset=args.dataset
    )
    para.update({"models": args.models.split('_')})
    # per_rel_eval(model_l, dataset=d, work_dir=para['work_dir'])
    pipeline_aggregation(d, para=para)
