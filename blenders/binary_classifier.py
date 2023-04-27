import argparse
import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from context_load_and_run import ContextLoader
from features.feature_per_rel_both_dataset import PerRelBothDataset
from features.feature_scores_only_dataset import ScoresOnlyDataset
from lp_kge.lp_pykeen import get_all_pos_triples
from blenders.blender_base import Blender, get_features_clz

if torch.cuda.is_available():
    import cuml as sk
    from cuml.svm import SVC
    from cuml.neural_network import MLPClassifier
else:
    import sklearn as sk
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier


names = [
    # "Nearest Neighbors",
    "LinearSVM",
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
    def __init__(self, params, logger):
        super().__init__(params, logger)
        self.context_loader = ContextLoader(in_dir=params.work_dir,
                                            model_list=params.models)

    def aggregate_scores(self):
        all_pos_triples = get_all_pos_triples(self.dataset)
        get_features = get_features_clz(self.params.features)
        test_data_feature = get_features(self.dataset.testing.mapped_triples, self.context_loader, all_pos_triples)
        if self.params.features in [2]:
            test_eval_feature, test_score_feature = torch.chunk(test_data_feature.get_all_test_examples(), 2, 1)
            pred_features = torch.mul(test_eval_feature, test_score_feature)
            dev_feature_dataset = PerRelBothDataset(self.dataset.validation.mapped_triples,
                                                    self.context_loader,
                                                    all_pos_triples,
                                                    feature=self.params.eval_feature,
                                                    num_neg=self.params.num_neg)
            pos, neg = dev_feature_dataset.get_all_dev_examples()
            ht_pos = torch.chunk(pos, 2, 1)
            ht_neg = torch.chunk(neg, 2, 1)
            pos = torch.mul(ht_pos[0], ht_pos[1])
            neg = torch.mul(ht_neg[0], ht_neg[1])

        else:
            dev_feature_dataset = ScoresOnlyDataset(self.dataset.validation.mapped_triples,
                                                    self.context_loader,
                                                    all_pos_triples,
                                                    num_neg=self.params.num_neg)
            pos, neg = dev_feature_dataset.get_all_dev_examples()
            pred_features = test_data_feature.get_all_test_examples()

        inputs = torch.cat([pos, neg], 0).numpy()
        labels = torch.cat([torch.ones(pos.shape[0]),
                            torch.zeros(neg.shape[0])], 0).numpy()
        for name, clf in zip(names, classifiers):
            clf.fit(inputs, labels)
            test_dataloader = DataLoader(pred_features.numpy(), batch_size=1000)
            individual_scores = []
            for batch in tqdm(test_dataloader):
                batch_scores = clf.predict_proba(batch.numpy())
                individual_scores.extend(batch_scores[:, 1])

            ht_blender = torch.as_tensor(individual_scores)
            option_str = f"{self.params.dataset}_{'_'.join(self.params.models)}_{name}"
            self.finalize(ht_blender, option_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="ComplEx_CP_RotatE_TuckER_anyburl")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument("--num_neg", type=int, default=4)
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    parser.add_argument('--eval_feature', type=str, default='rel')
    # "2": PerRelBothDataset,
    # "3": ScoresOnlyDataset,
    parser.add_argument('--features', type=int, default=2)  # 1, 2, 4, 6
    args = parser.parse_args()
    args.models = args.models.split('_')
    wab = BinaryClassifier(args, logging.getLogger(__name__))
    wab.aggregate_scores()
