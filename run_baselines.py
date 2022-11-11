import pykeen.datasets as ds
from pykeen.losses import CrossEntropyLoss
from pykeen.models import ComplEx, TuckER
from pykeen.pipeline import pipeline
from pykeen.evaluation import RankBasedEvaluator
from pykeen.regularizers import LpRegularizer
import logging

logging.basicConfig(level=logging.DEBUG)


def train_ComplEx(dataset):
    pipeline_result = pipeline(
        dataset=dataset,
        model="ComplEx",
        model_kwargs=dict(embedding_dim=512, entity_initializer="xavier_uniform",
                          relation_initializer="xavier_uniform"),
        loss=CrossEntropyLoss,
        loss_kwargs={"reduction": "mean"},
        regularizer=LpRegularizer,
        regularizer_kwargs=dict(weight=5e-2, p=3.0, ),
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
            "batch_size": 1024
        },
        result_tracker='mlflow',
        result_tracker_kwargs=dict(
            tracking_uri='http://127.0.0.1:5000',
            experiment_name='ComplEx training on FB237',
        ),
        stopper='early',
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
        evaluator_kwargs={"filtered": True},
        result_tracker='mlflow',
        result_tracker_kwargs=dict(
            tracking_uri='http://127.0.0.1:5000',
            experiment_name='TuckER training on FB237',
        ),
        stopper='early',
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
    )
    return pipeline_result


def train_multi_models(dataset, work_dir):
    for m in ['ComplEx', 'TuckER', 'RotatE']:
        func_name = 'train_' + m
        pipeline_result = globals()[func_name](dataset)
        pipeline_result.save_to_directory(work_dir+m+"/checkpoint/", save_metadata=True)


if __name__ == '__main__':
    # train_multi_models(dataset=ds.Nations(), work_dir="outputs/nations/")
    train_multi_models(dataset=ds.FB15k237(), work_dir="outputs/fb237/")
