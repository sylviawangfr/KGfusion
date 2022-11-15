import json

import pykeen
from pykeen.datasets import Nations, FB15k237
from pykeen.pipeline import pipeline
from pykeen.evaluation import RankBasedEvaluator
from pykeen.utils import prepare_filter_triples
from raw_score_evaluator import predict_head_tail_scores, per_rel_rank_evaluate
from score_blender import generate_training_input_feature, generate_pred_input_feature, train_linear_blender, \
    aggregate_scores
from context_load_and_run import per_rel_eval, load_score_context, test_multi_models_eval_individual


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


def get_all_pos_triples(dataset: pykeen.datasets.Dataset):
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
    context = test_multi_models_eval_individual(model_keywords=['TransE', 'ComplEx'], dataset=d)
    all_pos = get_all_pos_triples(d)
    p, n = generate_training_input_feature(d.validation.mapped_triples, context, all_pos)
    para = dict(lr=0.1, weight_decay=5e-3, epochs=3, models=['TransE', 'ComplEx'], dataset='Nations')
    mo = train_linear_blender(p.shape[1], p, n, params=para, work_dir="outputs/")
    re = aggregate_scores(mo, d.testing.mapped_triples, context, all_pos)
    print(re.dict())


def test_blender(dataset, work_dir, para):
    context = load_score_context(model_list, in_dir=work_dir)
    all_pos = get_all_pos_triples(dataset)
    p, n = generate_training_input_feature(d.validation.mapped_triples, context, all_pos)
    mo = train_linear_blender(p.shape[1], p, n, params=para, work_dir=work_dir)
    re = aggregate_scores(mo, d.testing.mapped_triples, context, all_pos)
    print(json.dumps(re.to_dict(), indent=2))
    json.dump(re.to_dict(), open(work_dir+"blended.json", "w"), indent=2)


if __name__ == '__main__':
    # test_lp()
    # test_small_dataset()
    model_list = ['ComplEx', 'TuckER', 'RotatE']

    # para = dict(lr=0.1, weight_decay=5e-3, epochs=3, models=model_list)
    # d = Nations()
    # wdr = 'outputs/nations/'

    para = dict(lr=1, weight_decay=5e-1, epochs=500, models=model_list)
    wdr = 'outputs/fb237/'
    d = FB15k237()

    # per_rel_eval(model_list, dataset=d, work_dir=wdr)
    test_blender(d, work_dir=wdr, para=para)


