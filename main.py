import pykeen
from pykeen.datasets import Nations
from pykeen.pipeline import pipeline
from pykeen.evaluation import RankBasedEvaluator
import pandas as pd
from pykeen.utils import prepare_filter_triples
from torch_geometric import datasets

from raw_score_evaluator import predict_head_tail_scores, per_rel_rank_evaluate
from score_blender import generate_training_input_feature, generate_pred_input_feature, train_linear_blender, \
    aggregate_scores
from context_load_and_run import multi_models_eval_individual, load_run_and_save, load_score_context


def test_lp():
    dataset = Nations()
    pipeline_result = pipeline(
        training=dataset.training,
        validation=dataset.validation,
        testing=dataset.testing,
        model='TransE',
        evaluator=RankBasedEvaluator,
        stopper='early',
        use_tqdm=False,
        # result_tracker='mlflow',
        # result_tracker_kwargs=dict(
        #     tracking_uri='http://127.0.0.1:5000',
        #     experiment_name='Tutorial Training of RotatE on Kinships',
        # ),
    )
    pipeline_result.save_to_directory('nations_transe', save_metadata=True)
    per_rel_eval = per_rel_rank_evaluate(pipeline_result.model, dataset.validation.mapped_triples)
    model_preds = predict_head_tail_scores(pipeline_result.model, dataset.testing.mapped_triples, mode=None)


def get_all_pos_triples(dataset:datasets):
    all_pos_triples = prepare_filter_triples(
        mapped_triples=dataset.testing.mapped_triples,
        additional_filter_triples=[
            dataset.training.mapped_triples,
            dataset.validation.mapped_triples,
        ],
    )
    return all_pos_triples


def test_small_dataset():
    d = Nations()
    context = multi_models_eval_individual(model_keywords=['TransE', 'ComplEx'], dataset=d)
    all_pos = get_all_pos_triples(d)
    p, n = generate_training_input_feature(d.validation.mapped_triples, context, all_pos)
    para = dict(lr=0.1, weight_decay=5e-3, epochs=3, models=['TransE', 'ComplEx'], dataset='Nations')
    mo = train_linear_blender(p.shape[1], p, n, params=para, work_dir="outputs/")
    re = aggregate_scores(mo, d.testing.mapped_triples, context, all_pos)
    print(re.dict())


def test_load_and_train():
    d = Nations()
    model_list = ['TuckER', 'ComplEx']
    work_dir = 'outputs/nations/'
    load_run_and_save(model_list, dataset=d, work_dir=work_dir)
    context = load_score_context(model_list, in_dir=work_dir)
    all_pos = get_all_pos_triples(d)
    p, n = generate_training_input_feature(d.validation.mapped_triples, context, all_pos)
    para = dict(lr=0.1, weight_decay=5e-3, epochs=3, models=model_list, dataset='Nations')
    mo = train_linear_blender(p.shape[1], p, n, params=para, work_dir=work_dir)
    re = aggregate_scores(mo, d.testing.mapped_triples, context, all_pos)
    print(re.dict())

if __name__ == '__main__':
    # test_lp()
    # test_small_dataset()
    test_load_and_train()


