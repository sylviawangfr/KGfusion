import argparse
from pathlib import Path

import torch
import pykeen.datasets as ds
from pykeen.datasets import ConceptNet, UMLS, get_dataset, WN18RR
from pykeen.losses import CrossEntropyLoss, BCEWithLogitsLoss
from pykeen.models import ComplEx, TuckER, NodePiece
from pykeen.pipeline import pipeline, replicate_pipeline_from_config, pipeline_from_config
from pykeen.evaluation import RankBasedEvaluator
from pykeen.regularizers import LpRegularizer
import logging

# logging.basicConfig(level=logging.DEBUG)
from pykeen.utils import normalize_path, load_configuration

from common_utils import init_dir


def train_TuckER(dataset):
    path = normalize_path(f"settings/TuckER_{dataset}.json")
    config = load_configuration(path)
    pipeline_result = pipeline_from_config(config=config)
    return pipeline_result


def train_ComnplEx(dataset):
    path = normalize_path(f"settings/ComplEx_{dataset}.json")
    config = load_configuration(path)
    pipeline_result = pipeline_from_config(config=config)
    return pipeline_result


def train_NodePiece(dataset):
    path = normalize_path("../galkin2022_nodepiece_fb15k237.yaml")
    config = load_configuration(path)
    config['pipeline']['dataset'] = dataset
    pipeline_results = pipeline_from_config(config=config)
    return pipeline_results


def train_RotatE(dataset):
    path = normalize_path(f"settings/RotatE_{dataset}.json")
    config = load_configuration(path)
    pipeline_result = pipeline_from_config(config=config)
    return pipeline_result


def train_multi_models(params):
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
    # parser.add_argument('--models', type=str, default="ComplEx_TuckER_RotatE")
    # parser.add_argument('--models', type=str, default="NodePiece")
    parser.add_argument('--models', type=str, default="TuckER")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    args = parser.parse_args()
    param1 = args.__dict__
    param1.update({"models": args.models.split('_')})
    # train_model("RotatE", "UMLS", param1['work_dir'])
    train_multi_models(param1)
    # train_NodePiece('fb15k23')
    # train_NodePiece('UMLS')
    # train_multi_models('fb15k237', ['NodePiece'], work_dir="outputs/fb237/")
