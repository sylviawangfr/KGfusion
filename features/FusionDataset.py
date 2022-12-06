import random

from pykeen.typing import MappedTriples
from torch.utils.data import Dataset
import torch
from abc import ABC, abstractmethod

from common_utils import chart_input


class FusionDataset(Dataset, ABC):
    def __init__(self, mapped_triples: MappedTriples, context_resource, all_pos_triples, num_neg=4):
        self.mapped_triples = mapped_triples
        self.num_triples = mapped_triples.shape[0]
        self.triple_index = list(range(mapped_triples.shape[0]))
        self.context_resource = context_resource
        self.releval2idx = context_resource['releval2idx']
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