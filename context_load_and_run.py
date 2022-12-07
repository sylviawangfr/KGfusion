import logging
import torch
from pykeen.datasets import Nations
import utils

logger = logging.getLogger(__name__)


def load_score_context(model_list, in_dir):
    context_resource = {m: {} for m in model_list}
    for m in model_list:
        read_dir = in_dir + m + '/'
        rel_eval = torch.load(read_dir + "rel_eval.pt")
        eval_pos_scores = torch.load(read_dir + "eval_pos_scores.pt")
        eval_neg_scores = torch.load(read_dir + "eval_neg_scores.pt")
        eval_neg_index = torch.load(read_dir + "eval_neg_index.pt")
        preds = torch.load(read_dir + "preds.pt")
        context_resource[m] = {'rel_eval': rel_eval,
                               'eval_pos_scores': eval_pos_scores,
                               'eval_neg_scores': eval_neg_scores,
                               'eval_neg_index': eval_neg_index,
                               'preds': preds}
    releval2idx = utils.load_json(in_dir + "releval2idx.json")
    releval2idx = {int(k): releval2idx[k] for k in releval2idx}
    context_resource.update({'releval2idx': releval2idx, 'models': model_list})
    return context_resource



