from pykeen.typing import MappedTriples
from torch.utils.data import Dataset
import torch
from abc import ABC, abstractmethod


class FusionDataset(Dataset, ABC):
    def __init__(self, mapped_triples: MappedTriples, context_resource, all_pos_triples, num_neg=4):
        self.mapped_triples = mapped_triples
        self.num_triples = mapped_triples.shape[0]
        self.triple_index = list(range(mapped_triples.shape[0]))
        self.context_resource = context_resource
        self.releval2idx = context_resource['releval2idx']
        self.all_pos_triples = all_pos_triples
        self.num_neg = num_neg
        self.train_pos = None
        self.train_neg = None
        self.test_examples = None

    def __getitem__(self, index):
        return self.triple_index[index]

    def __len__(self):
        return self.num_triples

    def collate_train(self, batch):
        if self.train_neg is None or self.train_pos is None:
            self.train_pos, self.train_neg = self.generate_training_input_feature()
            self.train_pos = torch.chunk(self.train_pos, self.num_triples, 0)
            self.train_neg = torch.chunk(self.train_neg, self.num_triples, 0)
        # fetch from pre-processed list
        batch_tensors_pos = torch.vstack(self.train_pos[batch])
        batch_tensors_neg = torch.vstack(self.train_neg[batch])
        return batch_tensors_pos, batch_tensors_neg

    def collate_test(self, batch):
        if self.test_examples is None:
            self.test_examples = self.generate_pred_input_feature()
            self.test_examples = torch.chunk(self.test_examples, 2, 0)
            self.test_examples = torch.cat(self.test_examples, 1)  # row: h + t
            candidate_num = int(self.test_examples.shape[0] / self.num_triples)
            self.test_examples = self.test_examples.reshape(self.num_triples, candidate_num,
                                                            self.test_examples.shape[-1])
        # fetch from pre-processed list: # row: h + t
        ht_batch_tensors = self.test_examples[torch.as_tensor(batch)]
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

