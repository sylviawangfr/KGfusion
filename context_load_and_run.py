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
        mapping_eval = torch.load(read_dir + "mapping_eval.pt")
        eval_pos_scores = torch.load(read_dir + "eval_pos_scores.pt")
        eval_neg_scores = torch.load(read_dir + "eval_neg_scores.pt")
        eval_neg_index = torch.load(read_dir + "eval_neg_index.pt")
        preds = torch.load(read_dir + "preds.pt")
        h_ent_eval = torch.load(read_dir + "h_ent_eval.pt")
        t_ent_eval = torch.load(read_dir + "t_ent_eval.pt")
        context_resource[m] = {'rel_eval': rel_eval,
                               'mapping_eval': mapping_eval,
                               'eval_pos_scores': eval_pos_scores,
                               'eval_neg_scores': eval_neg_scores,
                               'eval_neg_index': eval_neg_index,
                               'h_ent_eval': h_ent_eval,
                               't_ent_eval': t_ent_eval,
                               'preds': preds}
    releval2idx = utils.load_json(in_dir + "releval2idx.json")
    relmapping2idx = utils.load_json(in_dir + "relmapping2idx.json")
    h_ent2idx = utils.load_json(in_dir + "h_ent2idx.json")
    t_ent2idx = utils.load_json(in_dir + "t_ent2idx.json")
    releval2idx = {int(k): releval2idx[k] for k in releval2idx}
    context_resource.update({'releval2idx': releval2idx,
                             'relmapping2idx': relmapping2idx,
                             'h_ent2idx': h_ent2idx,
                             't_ent2idx': t_ent2idx,
                             'models': model_list})
    return context_resource



