import argparse
from analysis.degree_chart import EntDegreeChart
from analysis.rel_charts import RelMappingChart


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="CP_CPComplEx_TuckER_RotatE_anyburl")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    parser.add_argument('--cali', type=str, default="True")
    args = parser.parse_args()
    args.models = args.models.split('_')
    jobs = [EntDegreeChart, RelMappingChart]
    for j in jobs:
        tmp_job = j(args)
        tmp_job.analyze()
