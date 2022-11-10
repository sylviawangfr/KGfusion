import pykeen.datasets as ds
from pykeen.losses import CrossEntropyLoss
from pykeen.models import ComplEx
from pykeen.pipeline import pipeline
from pykeen.evaluation import RankBasedEvaluator
from pykeen.regularizers import LpRegularizer
import logging

logging.basicConfig(level=logging.DEBUG)


def test_linkprediction():
    pipeline_result = pipeline(
        dataset=ds.FB15k237(),
        model=ComplEx,
        model_kwargs=dict(embedding_dim=512, entity_initializer="xavier_uniform", relation_initializer="xavier_uniform"),
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
        # stopper='early',
    )
    pipeline_result.save_to_directory('fb15K237_complex')



if __name__ == '__main__':
    test_linkprediction()
