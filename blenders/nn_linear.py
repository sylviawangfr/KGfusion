import argparse
import logging
import os
import gc
import torch.nn.functional as F
from mlflow.entities import ViewType
from blenders.blender_base import Blender, get_features_clz
from pykeen.utils import resolve_device
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange
from mlflow import log_param
import mlflow.pytorch
from context_load_and_run import ContextLoader
from lp_kge.lp_pykeen import get_all_pos_triples

logging.basicConfig(level=logging.INFO)


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
    optimizer = torch.optim.SGD(params=model.parameters(), lr=params.lr, momentum=params.momentum)
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
    optimizer = torch.optim.SGD(params=model.parameters(), lr=params.lr, momentum=params.momentum)
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
    loss_func = nn.MarginRankingLoss(margin=params.margin, reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
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


def get_MLP(keyword='2'):
    clz = {"1": ScoreBlenderLinear1,
           "2": ScoreBlenderLinear2,
           "3": ScoreBlenderLinear3}
    return clz[keyword]


class NNLinearBlender(Blender):
    def __init__(self, params, logger):
        super().__init__(params, logger)
        self.all_pos_triples = get_all_pos_triples(self.dataset)
        self.context_loader = ContextLoader(in_dir=params.work_dir, model_list=params.models)
        self.params.dataset = self.dataset

    def _train_aggregation_model(self):
        dataset_cls = get_features_clz(self.params.features)
        train_data = dataset_cls(self.dataset.validation.mapped_triples,
                                 self.context_loader,
                                 self.all_pos_triples,
                                 calibrated=False,
                                 num_neg=self.params.num_neg)
        train_dataloader = DataLoader(train_data, batch_size=self.params.batch_size, shuffle=True, collate_fn=train_data.collate_train)
        train_loop = get_train_loop(self.params.loss)
        num_epochs = self.params.epochs
        epochs = trange(num_epochs)
        model_cls = get_MLP(self.params.linear)
        model = model_cls(train_data.dim)
        device: torch.device = resolve_device()
        # logger.info(f"Using device: {device}")
        self.logger.info(f"Train Using device: {device}")
        if device is not None:
            # logger.info(f"Send model to device: {device}")
            self.logger.info(f"Send model to device: {device}")
            model.to(device)
        model.train()
        for e in epochs:
            train_loss = train_loop(model, train_dataloader, device, self.params)
            epochs.set_postfix_str({"loss: {}".format(train_loss)})
            if self.params.mlflow:
                mlflow.log_metric("train_loss", train_loss, step=e)
        work_dir = self.params.work_dir
        torch.save(model, os.path.join(work_dir,
                                       'margin_ensemble.pth'))
        return model

    def _nn_aggregate_scores(self, model):
        # Send to device
        device: torch.device = resolve_device()
        self.logger.info(f"Eval Using device: {device}")
        if device is not None and next(model.parameters()).device != device:
            model.to(device)
        # Ensure evaluation mode
        model.eval()
        # Send tensors to device
        h_preds = []
        t_preds = []
        dataloader_cls = get_features_clz(self.params.features)
        test_per_rel_dataset = dataloader_cls(self.dataset.testing.mapped_triples,
                                              self.context_loader,
                                              self.all_pos_triples,
                                              calibrated=True)
        test_dataloader = DataLoader(test_per_rel_dataset, batch_size=32, collate_fn=test_per_rel_dataset.collate_test)
        for i, batch in enumerate(test_dataloader):
            ens_logits = model(batch.to(device))
            ht_preds = torch.chunk(ens_logits, 2, 0)
            h_preds.append(ht_preds[0])
            t_preds.append(ht_preds[1])
        # restore format that required by pykeen evaluator
        h_preds = torch.cat(h_preds, 0)
        t_preds = torch.cat(t_preds, 0)
        ht_preds = torch.cat([h_preds, t_preds], 0)
        return ht_preds.detach().cpu()

    def aggregate_scores(self):
        self.logger.info("testing")
        if self.params.mlflow:
            exp = mlflow.search_experiments(view_type=ViewType.ALL, filter_string="name='KGFusion'")
            if not exp:
                mlflow.create_experiment("KGFusion")
            mlflow.set_experiment("KGFusion")
            log_para = self.params.__dict__
            for k in log_para:
                log_param(k, log_para[k])
        # 2. train model
        mo = self._train_aggregation_model()
        # 3. aggregate scores for testing
        ht_scores = self._nn_aggregate_scores(mo)

        option_str = f"{self.dataset.metadata['name']}_" \
                     f"{'_'.join(self.params.models)}_" \
                     f"data{self.params.features}_" \
                     f"{self.params.loss}_nn"
        self.finalize(ht_scores, option_str)


if __name__ == '__main__':
    # features
    # 1: PerRelDataset,
    # 2: PerRelBothDataset,
    # 3: ScoresOnlyDataset,
    # 4: PerEntDataset,
    # 5: PerRelEntDataset
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="ComplEx_CP_RotatE_TuckER_anyburl")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument('--features', type=int, default=3)
    parser.add_argument('--eval_feature', type=str, default='rel')
    parser.add_argument('--linear', type=str, default="1")
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--loss", type=str, default='bce')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_neg", type=int, default=5)
    parser.add_argument("--mlflow", type=str, default='True')
    parser.add_argument("--momentum", type=float, default=0.8)
    parser.add_argument("--margin", type=float, default=5)
    args = parser.parse_args()
    if args.mlflow == 'True':
        args.mlflow = True
    args.models = args.models.split('_')
    nnb = NNLinearBlender(args, logging.getLogger('NNLinearBlender'))
    nnb.aggregate_scores()
