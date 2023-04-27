import argparse
from analysis.ent_degree_chart import EntDegreeChart
from analysis.rel_chart import RelChart
from analysis.rel_mapping_chart import RelMappingChart


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="CP_ComplEx_TuckER_RotatE_anyburl")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    parser.add_argument('--cali', type=str, default="True")
    parser.add_argument('--num_p', type=int, default=10)
    args = parser.parse_args()
    args.models = args.models.split('_')
    jobs = [RelChart, RelMappingChart, EntDegreeChart]
    for j in jobs:
        tmp_job = j(args)
        key2triids = tmp_job.make_triple_partitions(tmp_job.dataset.validation.mapped_triples)
        tmp_job.partition_eval_and_save(key2triids)


