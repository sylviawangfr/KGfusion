import logging
import torch
import common_utils


logger = logging.getLogger(__name__)


def load_score_context(model_list, in_dir, calibration=False, evaluator_key=None, eval_feature=None):
    if eval_feature == "rel_mapping":
        eval_filename = "mapping_rel_eval.pt"
        releval2idx_filename = "mapping_releval2idx.json"
    elif eval_feature == "rel":
        eval_filename = "rel_eval.pt"
        releval2idx_filename = "releval2idx.json"
    else:
        eval_filename = "total_eval.pt"
    context_resource = {m: {} for m in model_list}
    context_resource.update({'models': model_list})
    for m in model_list:
        read_dir = in_dir + m + '/'
        eval_pos_scores = torch.load(read_dir + "eval_pos_scores.pt")
        eval_neg_scores = torch.load(read_dir + "eval_neg_scores.pt")
        eval_neg_index = torch.load(read_dir + "eval_neg_index.pt")
        if calibration:
            preds = torch.load(read_dir + "cali_preds.pt")
        else:
            preds = torch.load(read_dir + "preds.pt")
        context_resource[m] = {'eval_pos_scores': eval_pos_scores,
                               'eval_neg_scores': eval_neg_scores,
                               'eval_neg_index': eval_neg_index,
                               'preds': preds}

        if evaluator_key is not None:
            if eval_feature == "total":
                rel_eval = torch.load(read_dir + f"{evaluator_key}_{eval_filename}")
                context_resource[m].update({'total_eval': rel_eval,
                                            })
            else:
                rel_eval = torch.load(read_dir + f"{evaluator_key}_{eval_filename}")
                # h_ent_eval = torch.load(read_dir + f"{evaluator_key}_h_ent_eval.pt")
                # t_ent_eval = torch.load(read_dir + f"{evaluator_key}_t_ent_eval.pt")
                context_resource[m].update({'rel_eval': rel_eval,
                                                # 'h_ent_eval': h_ent_eval,
                                                # 't_ent_eval': t_ent_eval
                                                })
    if evaluator_key is not None and eval_feature != "total":
        releval2idx = common_utils.load_json(in_dir + f"{evaluator_key}_{releval2idx_filename}")
        # h_ent2idx = common_utils.load_json(in_dir + f"{evaluator_key}_h_ent2idx.json")
        # t_ent2idx = common_utils.load_json(in_dir + f"{evaluator_key}_t_ent2idx.json")
        releval2idx = {int(k): releval2idx[k] for k in releval2idx}
        # h_ent2idx = {int(k): h_ent2idx[k] for k in h_ent2idx}
        # t_ent2idx = {int(k): t_ent2idx[k] for k in t_ent2idx}
        context_resource.update({'releval2idx': releval2idx,
                                 # 'h_ent2idx': h_ent2idx,
                                 # 't_ent2idx': t_ent2idx
                                 })
    return context_resource
