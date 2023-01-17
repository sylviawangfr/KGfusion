import argparse
from pathlib import Path

import torch
import pykeen.datasets as ds
from pykeen.datasets import ConceptNet, UMLS, get_dataset
from pykeen.losses import CrossEntropyLoss, BCEWithLogitsLoss
from pykeen.models import ComplEx, TuckER, NodePiece
from pykeen.pipeline import pipeline, replicate_pipeline_from_config, pipeline_from_config
from pykeen.evaluation import RankBasedEvaluator
from pykeen.regularizers import LpRegularizer
import logging

# logging.basicConfig(level=logging.DEBUG)
from pykeen.utils import normalize_path, load_configuration

from common_utils import init_dir


def train_ComplEx2(dataset):
    pipeline_result = pipeline(
        dataset=dataset,
        model="ComplEx",
        model_kwargs=dict(embedding_dim=512, entity_initializer="xavier_uniform",
                          relation_initializer="xavier_uniform"),
        loss=CrossEntropyLoss,
        loss_kwargs={"reduction": "mean"},
        regularizer=LpRegularizer,
        regularizer_kwargs=dict(weight=5e-2, p=2.0, ),
        # lr_scheduler='ExponentialLR',
        # lr_scheduler_kwargs=dict(gamma=0.99, ),
        optimizer="adagrad",
        optimizer_kwargs=dict(lr=0.5),
        evaluator=RankBasedEvaluator,
        training_loop="SLCWA",
        negative_sampler="basic",
        negative_sampler_kwargs={"num_negs_per_pos": 10},
        training_kwargs={
            "num_epochs": 1000,
            # "num_epochs": 10,
            "batch_size": 1024
        },
        # result_tracker='mlflow',
        # result_tracker_kwargs=dict(
        #     tracking_uri='http://127.0.0.1:5000',
        #     experiment_name='ComplEx training on FB237',
        # ),
        stopper='early',
        stopper_kwargs={"patience": 10},
        evaluator_kwargs={"filtered": True}
    )
    return pipeline_result


def train_ComplEx(dataset):
    pipeline_result = pipeline(
        dataset=dataset,
        model="ComplEx",
        model_kwargs=dict(embedding_dim=200, entity_initializer="xavier_uniform",
                          relation_initializer="xavier_uniform", predict_with_sigmoid=True),
        loss=CrossEntropyLoss,
        loss_kwargs={"reduction": "mean"},
        regularizer=LpRegularizer,
        regularizer_kwargs=dict(weight=0.01, p=2.0, ),
        # lr_scheduler='ExponentialLR',
        # lr_scheduler_kwargs=dict(gamma=0.99, ),
        optimizer="adagrad",
        optimizer_kwargs=dict(lr=0.5),
        evaluator=RankBasedEvaluator,
        training_loop="SLCWA",
        negative_sampler="basic",
        negative_sampler_kwargs={"num_negs_per_pos": 10},
        training_kwargs={
            "num_epochs": 1000,
            # "num_epochs": 10,
            "batch_size": 1024
        },
        # result_tracker='mlflow',
        # result_tracker_kwargs=dict(
        #     tracking_uri='http://127.0.0.1:5000',
        #     experiment_name='ComplEx training on FB237',
        # ),
        stopper='early',
        stopper_kwargs={"patience": 10},
        evaluator_kwargs={"filtered": True}
    )
    return pipeline_result


def train_TuckER(dataset):
    pipeline_result = pipeline(
        dataset=dataset,
        dataset_kwargs={"create_inverse_triples": True},
        model="TuckER",
        model_kwargs={
            "embedding_dim": 200,
            "relation_dim": 200,
            "dropout_0": 0.3,
            "dropout_1": 0.4,
            "dropout_2": 0.5,
            "apply_batch_normalization": True,
            "entity_initializer": "xavier_normal",
            "relation_initializer": "xavier_normal"
        },
        optimizer="Adam",
        optimizer_kwargs={"lr": 0.0005},
        lr_scheduler="ExponentialLR",
        lr_scheduler_kwargs={"gamma": 1.0},
        loss="BCEAfterSigmoid",
        loss_kwargs={"reduction": "mean"},
        training_loop="LCWA",
        training_kwargs={
            "num_epochs": 500,
            "batch_size": 128,
            "label_smoothing": 0.1
        },
        # evaluator_kwargs={"filtered": True},
        # result_tracker='mlflow',
        # result_tracker_kwargs=dict(
        #     tracking_uri='http://127.0.0.1:5000',
        #     experiment_name='TuckER training on FB237',
        # ),
        stopper='early',
        stopper_kwargs={"patience": 10}
    )
    return pipeline_result


def train_RotatE(dataset):
    pipeline_result = pipeline(
        dataset=dataset,
        model="RotatE",
        model_kwargs={
            "embedding_dim": 1000,
            "entity_initializer": "uniform",
            "relation_initializer": "init_phases",
            "relation_constrainer": "complex_normalize"
        },
        optimizer="Adam",
        optimizer_kwargs={
            "lr": 0.00005
        },
        loss="nssa",
        loss_kwargs={
            "reduction": "mean",
            "adversarial_temperature": 1.0,
            "margin": 9
        },
        training_loop="SLCWA",
        negative_sampler="basic",
        negative_sampler_kwargs={
            "num_negs_per_pos": 256
        },
        training_kwargs={
            "num_epochs": 1000,
            "batch_size": 1024
        },
        evaluator_kwargs={
            "filtered": True
        },
        stopper='early',
        stopper_kwargs={"patience": 10},
        # result_tracker='mlflow',
        # result_tracker_kwargs=dict(
        #     tracking_uri='http://127.0.0.1:5000',
        #     experiment_name='RotatE training on FB237',
        # ),
    )
    return pipeline_result


def train_NodePiece(dataset):
    # extra_kwargs = dict(move_to_cpu=False,
    #                     save_replicates=False,
    #                     save_training=False)
    path = normalize_path("galkin2022_nodepiece_fb15k237.yaml")
    config = load_configuration(path)
    config['pipeline']['dataset'] = dataset
    pipeline_results = pipeline_from_config(config=config)
    return pipeline_results


class DeepSet(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, dim=-2):
        x = self.encoder(x).mean(dim)
        x = self.decoder(x)
        return x


# def train_NodePiece2(dataset):
#     result = pipeline(
#         dataset=dataset,
#         dataset_kwargs=dict(
#             create_inverse_triples=True,
#         ),
#         model=NodePiece,
#         model_kwargs=dict(
#             tokenizers=["AnchorTokenizer", "RelationTokenizer"],
#             num_tokens=[20, 12],
#             tokenizers_kwargs=[
#                 dict(
#                     selection="MixtureAnchorSelection",
#                     selection_kwargs=dict(
#                         selections=["degree", "pagerank", "random"],
#                         ratios=[0.4, 0.4, 0.2],
#                         num_anchors=500,
#                     ),
#                     searcher="ScipySparse",
#                 ),
#                 dict(),  # empty dict for the RelationTokenizer - it doesn't need any kwargs
#             ],
#             embedding_dim=64,
#             interaction="rotate",
#             relation_initializer="init_phases",
#             relation_constrainer="complex_normalize",
#             entity_initializer="xavier_uniform_",
#             aggregation=DeepSet(hidden_dim=64),
#         ),
#         loss=BCEWithLogitsLoss,
#         loss_kwargs=dict(reduction='mean'),
#         optimizer="Adam",
#         optimizer_kwargs=dict(
#             lr=0.0005),
#         training_kwargs=dict(
#                 batch_size=512,
#                 num_epochs=401,
#                 label_smoothing=0.4),
#         training_loop='LCWA'
#     )
#     return result


def train_multi_models(params):
    # dataset = get_dataset(
    #     dataset=params['dataset']
    # )
    dataset = params['dataset']
    model_list = params['models']
    work_dir = params['work_dir']
    init_dir(work_dir)
    for m in model_list:
        func_name = 'train_' + m
        pipeline_result = globals()[func_name](dataset)
        pipeline_result.save_to_directory(work_dir + m + "/checkpoint/", save_metadata=True)
        # if 'NodePiece' in m:
        #     pipeline_result.entity_representations[0].base[0].save_assignment(
        #         Path(work_dir + m + "/checkpoint/anchors_assignment.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    # parser.add_argument('--models', type=str, default="ComplEx_TuckER_RotatE_NodePiece")
    parser.add_argument('--models', type=str, default="NodePiece")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument('--work_dir', type=str, default="outputs/umls/")
    args = parser.parse_args()
    param1 = args.__dict__
    param1.update({"models": args.models.split('_')})
    train_multi_models(param1)
    # train_NodePiece('fb15k23')
    # train_NodePiece('UMLS')
    # train_multi_models('fb15k237', ['NodePiece'], work_dir="outputs/fb237/")
