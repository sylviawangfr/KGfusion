from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import random
import pandas as pd
from pykeen.typing import MappedTriples
import torch
from collections import Counter


class FusionDataset(Dataset, ABC):
    def __init__(self, mapped_triples: MappedTriples, context_resource, all_pos_triples, num_neg=4):
        self.mapped_triples = mapped_triples
        self.num_triples = mapped_triples.shape[0]
        self.triple_index = list(range(mapped_triples.shape[0]))
        self.context_resource = context_resource
        self.all_pos_triples = all_pos_triples
        self.num_neg = num_neg
        self._train_pos = None
        self._train_neg = None
        self._test_examples = None
        self._all_test_examples = None
        self._all_pos = None
        self._all_neg=None

    def __getitem__(self, index):
        return self.triple_index[index]

    def __len__(self):
        return self.num_triples

    def get_all_dev_examples(self):
        if self._all_pos is None or self._all_neg is None:
            self._all_pos, self._all_neg = self.generate_training_input_feature()
        return self._all_pos, self._all_neg

    def get_all_test_examples(self):
        if self._all_test_examples is None:
            self._all_test_examples = self.generate_pred_input_feature()
        return self._all_test_examples

    def collate_train(self, batch):
        if self._train_neg is None or self._train_pos is None:
            if self._all_pos is None:
                self._all_pos, self._all_neg = self.generate_training_input_feature()
            self._train_pos = self._all_pos.reshape(self.num_triples, int(self._all_pos.shape[0] / self.num_triples), self._all_pos.shape[-1])
            self._train_neg = self._all_neg.reshape(self.num_triples, int(self._all_neg.shape[0] / self.num_triples), self._all_neg.shape[-1])
        # fetch from pre-processed list
        batch_tensors_pos = self._train_pos[torch.as_tensor(batch)]
        batch_tensors_pos = batch_tensors_pos.reshape(batch_tensors_pos.shape[0] * batch_tensors_pos.shape[1], batch_tensors_pos.shape[2])
        batch_tensors_neg = self._train_neg[torch.as_tensor(batch)]
        batch_tensors_neg = batch_tensors_neg.reshape(batch_tensors_neg.shape[0] * batch_tensors_neg.shape[1], batch_tensors_neg.shape[2])
        return batch_tensors_pos, batch_tensors_neg

    def collate_test(self, batch):
        if self._test_examples is None:
            self._all_test_examples = self.generate_pred_input_feature()
            self._test_examples = torch.chunk(self._all_test_examples, 2, 0)
            self._test_examples = torch.cat(self._test_examples, 1)  # row: h + t
            candidate_num = int(self._test_examples.shape[0] / self.num_triples)
            self._test_examples = self._test_examples.reshape(self.num_triples, candidate_num,
                                                              self._test_examples.shape[-1])
        # fetch from pre-processed list: # row: h + t
        ht_batch_tensors = self._test_examples[torch.as_tensor(batch)]
        ht_batch_tensors = torch.chunk(ht_batch_tensors, 2, -1)
        ht_batch_tensors = torch.cat(ht_batch_tensors, 0)
        preserve_shape = ht_batch_tensors.shape
        ht_batch_tensors = ht_batch_tensors.reshape(preserve_shape[0] * preserve_shape[1], preserve_shape[2])
        return ht_batch_tensors

    @abstractmethod
    def generate_training_input_feature(self):
        pass

    @abstractmethod
    def generate_pred_input_feature(self):
        pass


def padding_sampling(x, expected_len):
    if len(x) < expected_len:
        for i in range(expected_len - len(x)):
            x.extend(random.sample(x, 1))
    return x


def get_multi_model_neg_topk(neg_score_ht, neg_index_topk, num_neg):
    selected_scores = []
    neg_scores = torch.stack(neg_score_ht, 0)
    neg_scores = torch.chunk(neg_scores, 2, 2)  # (h, t) tuple
    neg_index = torch.stack(neg_index_topk, 0)
    neg_index = torch.chunk(neg_index, 2, 2)
    #   [e1,h,e3,..],r,t
    #   h,r,[t,e2,...]
    #   The neg examples contain prediction scores and head/tail eval scores
    # we count the most frequent top_k examples from head and tail preds
    for target in range(2):
        # restore model scores and index to orginal indexed tensor in shape of [num_model, num_triples, num_candidates + 1]
        # this is an important step to gather scores from multi-models.
        topk_idx = neg_index[target].squeeze(2)
        max_index = torch.max(topk_idx)  # number of original candidates
        tmp_scores = neg_scores[target].squeeze(2)
        tmp_topk = torch.clone(topk_idx)
        # add one extra column to handle the -999.0/-1 mask values. the masked values are not selected.
        tmp_topk[tmp_topk == -1] = max_index + 1
        # restore scores to entity id order, the max x dim is the number of candidate entities
        scattered_scores = torch.zeros([tmp_topk.shape[0], tmp_topk.shape[1], max_index + 2]).scatter_(2, tmp_topk, tmp_scores) # sigmoid scores to [0-1]
        # target_neg_scores = neg_scores[target].squeeze().transpose(0,1).transpose(1,2)
        scattered_target_neg_scores = scattered_scores.transpose(0, 1).transpose(1, 2)
        # count top_k frequent index
        backup_shape = topk_idx.shape
        topk_idx = topk_idx.transpose(0, 1).reshape([backup_shape[1], backup_shape[0]*backup_shape[2]])
        idx_df = pd.DataFrame(data=topk_idx.numpy().T)
        idx_df = idx_df.groupby(idx_df.columns, axis=1).apply(lambda x: x.values).apply(lambda y:y.flatten())
        idx_df = idx_df.apply(lambda x: x[x != -1]).apply(lambda x: [a[0] for a in Counter(x).most_common(num_neg)])
        # for rows that less than num_neg, we randomly duplicate existing index
        idx_df = idx_df.apply(lambda x: padding_sampling(x, num_neg))
        tmp_topk_idx = list(idx_df.values)
        padded_topk = torch.as_tensor(tmp_topk_idx)
        target_neg_scores = scattered_target_neg_scores[torch.arange(0, scattered_target_neg_scores.shape[0]).unsqueeze(1), padded_topk]
        selected_scores.append(target_neg_scores)

    scores_frequent = torch.cat(selected_scores, 1)
    # random select from  top_k * times
    sample_weight = torch.ones(scores_frequent.shape[0], scores_frequent.shape[1])
    sample_index = torch.multinomial(sample_weight, num_neg)
    scores_frequent = scores_frequent[torch.arange(0, scores_frequent.shape[0]).unsqueeze(1), sample_index]
    selected_feature = scores_frequent.reshape(scores_frequent.shape[0] * scores_frequent.shape[1], scores_frequent.shape[2])
    return selected_feature