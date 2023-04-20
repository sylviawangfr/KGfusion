import argparse
import logging
import torch
from pykeen.evaluation import RankBasedEvaluator
from pykeen.typing import LABEL_HEAD, LABEL_TAIL
from torch.utils.data import DataLoader
from tqdm import tqdm
from context_load_and_run import load_score_context
from features.feature_per_rel_both_dataset import PerRelBothDataset
from features.feature_scores_only_dataset import ScoresOnlyDataset
from lp_kge.lp_pykeen import get_all_pos_triples
from blender_utils import eval_with_blender_scores, Blender, get_features_clz
from common_utils import format_result, save_to_file
if torch.cuda.is_available():
    import cuml as sk
    from cuml.svm import SVC
    from cuml.neural_network import MLPClassifier
else:
    import sklearn as sk
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier


logger = logging.getLogger(__name__)

names = [
    # "Nearest Neighbors",
    "Linear SVM",
    # "RBF SVM",
    # "Gaussian Process",
    # "DecisionTree",
    # "RandomForest",
    "NeuralNet",
    # "AdaBoost",
    # "NaiveBayes",
    # "QDA",
]

classifiers = [
    # KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, probability=True),
    # SVC(gamma=2, C=1, probability=True),
    # GaussianProcessClassifier(1.0 * RBF(1.0)),
    # DecisionTreeClassifier(max_depth=5),
    # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=500),
    # AdaBoostClassifier(),
    # GaussianNB(),
    # QuadraticDiscriminantAnalysis(),
]


class BinaryClassifier(Blender):
    def __init__(self, params):
        super().__init__(params)
        self.context = load_score_context(params.models,
                                          in_dir=params.work_dir,
                                          calibration=True,
                                          evaluator_key='rank',
                                          eval_feature=params.eval_feature,
                                          )

    def aggregate_scores(self):
        all_pos_triples = get_all_pos_triples(self.dataset)
        work_dir = self.params.work_dir
        models_context = self.context

        get_features = get_features_clz(self.params.features)
        test_data_feature = get_features(self.dataset.testing.mapped_triples, self.context, all_pos_triples)
        if self.params.features in [1, 2]:
            test_eval_feature, test_score_feature = torch.chunk(test_data_feature.get_all_test_examples(), 2, 1)
            pred_features = torch.mul(test_eval_feature, test_score_feature)
            dev_feature_dataset = PerRelBothDataset(self.dataset.validation.mapped_triples,
                                                    models_context,
                                                    all_pos_triples,
                                                    num_neg=self.params.num_neg)
            pos, neg = dev_feature_dataset.get_all_dev_examples()
            ht_pos = torch.chunk(pos, 2, 1)
            ht_neg = torch.chunk(neg, 2, 1)
            pos = torch.mul(ht_pos[0], ht_pos[1])
            neg = torch.mul(ht_neg[0], ht_neg[1])

        else:
            dev_feature_dataset = ScoresOnlyDataset(self.dataset.validation.mapped_triples,
                                                    models_context,
                                                    all_pos_triples,
                                                    num_neg=self.params.num_neg)
            pos, neg = dev_feature_dataset.get_all_dev_examples()
            pred_features = test_data_feature.get_all_test_examples()

        inputs = torch.cat([pos, neg], 0).numpy()
        labels = torch.cat([torch.ones(pos.shape[0]),
                            torch.zeros(neg.shape[0])], 0).numpy()
        # rate = pos.shape[0] / neg.shape[0]
        # weights = torch.cat([torch.ones(pos.shape[0]),
        #                      torch.full([neg.shape[0]], rate)]).numpy()

        for name, clf in zip(names, classifiers):
            # if name in ["AdaBoost", "NaiveBayes"]:
            #     clf.fit(inputs, labels, weights)
            # else:
            clf.fit(inputs, labels)
            test_dataloader = DataLoader(pred_features.numpy(), batch_size=1000)
            individual_scores = []
            for batch in tqdm(test_dataloader):
                batch_scores = clf.predict_proba(batch.numpy())
                individual_scores.extend(batch_scores[:, 1])
            h_preds, t_preds = torch.chunk(torch.as_tensor(individual_scores), 2, 0)
            # restore format that required by pykeen evaluator
            candidate_number = self.dataset.num_entities
            ht_scores = [h_preds.reshape([self.dataset.testing.num_triples, candidate_number]),
                         t_preds.reshape([self.dataset.testing.num_triples, candidate_number])]
            evaluator = RankBasedEvaluator()
            relation_filter = None
            for ind, target in enumerate([LABEL_HEAD, LABEL_TAIL]):
                relation_filter = eval_with_blender_scores(
                    batch=self.dataset.testing.mapped_triples,
                    scores=ht_scores[ind],
                    target=target,
                    evaluator=evaluator,
                    all_pos_triples=all_pos_triples,
                    relation_filter=relation_filter,
                )
            result = evaluator.finalize()
            str_re = format_result(result)
            option_str = f"{self.params.dataset}_{'_'.join(self.params.models)}_{name}"
            save_to_file(str_re, work_dir + f"{option_str}.log")
            print(f"{option_str}:\n{str_re}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="anyburl_CPComplEx")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument("--num_neg", type=int, default=1)
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    parser.add_argument('--eval_feature', type=str, default='rel')
    # "1": PerRelDataset,
    # "2": PerRelBothDataset,
    # "3": ScoresOnlyDataset,
    parser.add_argument('--features', type=int, default=3)  # 1, 2, 4, 6
    args = parser.parse_args()
    args.models = args.models.split('_')
    wab = BinaryClassifier(args)
    wab.aggregate_scores()
