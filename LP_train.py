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


def multi_models_eval_individual(model_keywords: [], dataset: pykeen.datasets, work_dir=''):
    context_resource = {m: {} for m in model_keywords}
    # group by relations
    mapped_triples_eval = dataset.validation.mapped_triples
    triples_df = pd.DataFrame(data=mapped_triples_eval.numpy(), columns=['h', 'r', 't'])
    groups = triples_df.groupby('r', group_keys=True, as_index=False)
    eval_group2idx = {key: idx for idx, key in enumerate(groups.groups.keys())}
    ordered_keys = eval_group2idx.keys()
    for m in model_keywords:
        single_model = pipeline(
            training=dataset.training,
            validation=dataset.validation,
            testing=dataset.testing,
            model=m,
            evaluator=RankBasedEvaluator,
            stopper='early'
        )
        per_rel_eval = per_rel_rank_evaluate(single_model.model, groups, ordered_keys)
        eval_preds = predict_head_tail_scores(single_model.model, dataset.validation.mapped_triples, mode=None)
        test_preds = predict_head_tail_scores(single_model.model, dataset.testing.mapped_triples, mode=None)
        context_resource[m] = {'model': single_model, 'rel_eval': per_rel_eval, 'eval': eval_preds, 'preds': test_preds}
    context_resource.update({'releval2idx': eval_group2idx, 'models': model_keywords})
    return context_resource


def get_all_pos_triples(dataset:datasets):
    all_pos_triples = prepare_filter_triples(
        mapped_triples=dataset.testing.mapped_triples,
        additional_filter_triples=[
            dataset.training.mapped_triples,
            dataset.validation.mapped_triples,
        ],
    )
    return all_pos_triples


if __name__ == '__main__':
    # test_lp()
    d = Nations()
    context = multi_models_eval_individual(model_keywords=['TransE', 'ComplEx'], dataset=d)
    all_pos = get_all_pos_triples(d)
    p, n = generate_training_input_feature(d.validation.mapped_triples, context, all_pos)
    para = dict(lr=0.1, weight_decay=5e-3, epochs=3, models=['TransE', 'ComplEx'], dataset='Nations')
    mo = train_linear_blender(p.shape[1], p, n, params=para, work_dir="outputs/")
    re = aggregate_scores(mo, d.testing.mapped_triples, context, all_pos)
    print(re.dict())