import pandas as pd
from pykeen.datasets import FB15k237, UMLS, get_dataset
import pykeen.datasets
from common_utils import save_to_file, wait_until_file_is_saved, init_dir
import os.path as osp


def mapped_triples_2_kbc(dataset: pykeen.datasets.Dataset, src_data_dir):
    init_dir(src_data_dir)
    df_train = pd.DataFrame(dataset.training.mapped_triples.numpy()).astype('int64')
    df_train.to_csv(osp.join(src_data_dir, f'train'), header=False, index=False, sep='\t')
    df_dev = pd.DataFrame(dataset.validation.mapped_triples.numpy()).astype('int64')
    df_dev.to_csv(osp.join(src_data_dir, f'valid'), header=False, index=False, sep='\t')
    # use dev as blender training set
    df_test = pd.DataFrame(dataset.testing.mapped_triples.numpy()).astype('int64')
    df_test.to_csv(osp.join(src_data_dir, f'test'), header=False, index=False, sep='\t')


if __name__ == "__main__":
    mapped_triples_2_kbc(UMLS(), "src_data/UMLS/")
    mapped_triples_2_kbc(FB15k237(), "src_data/FB15k237/")