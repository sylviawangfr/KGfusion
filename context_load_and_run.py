import logging
import torch
import common_utils

logger = logging.getLogger(__name__)


class ContextLoader:
    def __init__(self, in_dir, model_list):
        self.in_dir = in_dir
        self.context_resource = {m: {} for m in model_list}
        self.models = model_list
        self.cache_preds = False


    def set_cache_preds(self, signal: bool):
        self.cache_preds = signal
        return self

    def load_preds(self, model_list, calibrated=False):
        tmp_m2eval = {m: {} for m in model_list} if not self.cache_preds else self.context_resource
        for m in model_list:
            read_dir = self.in_dir + m + '/'
            if calibrated:
                preds = torch.load(read_dir + "cali_preds.pt")
            else:
                preds = torch.load(read_dir + "preds.pt")
            tmp_m2eval[m].update({'preds': preds})
        return tmp_m2eval

    def load_valid_preds(self, model_list, cache=False):
        tmp_m2eval = {m: {} for m in model_list} if not cache else self.context_resource
        for m in model_list:
            read_dir = self.in_dir + m + '/'
            evals = torch.load(read_dir + "valid_preds.pt")
            tmp_m2eval[m].update({'valid_preds': evals})
        return tmp_m2eval

    def load_eval_examples(self, model_list, cache=False):
        tmp_m2eval = {m: {} for m in model_list} if not cache else self.context_resource
        for m in model_list:
            read_dir = self.in_dir + m + '/'
            eval_pos_scores = torch.load(read_dir + "eval_pos_scores.pt")
            eval_neg_scores = torch.load(read_dir + "eval_neg_scores.pt")
            eval_neg_index = torch.load(read_dir + "eval_neg_index.pt")
            tmp_m2eval[m].update({'eval_pos_scores': eval_pos_scores,
                                             'eval_neg_scores': eval_neg_scores,
                                             'eval_neg_index': eval_neg_index})
        return tmp_m2eval

    def load_rel_eval(self, model_list, cache=False):
        tmp_m2eval = {m: {} for m in model_list} if not cache else self.context_resource
        for m in model_list:
            file_name = self.in_dir + f'rel_eval/{m}_rel_eval.pt'
            rel_eval = torch.load(file_name)
            tmp_m2eval[m].update({'rel_eval': rel_eval})
        rel2eval_idx = common_utils.load_json(self.in_dir + f"rel_eval/rel2eval_idx.json",
                                              object_hook=common_utils.jsonkeys2int)
        tmp_m2eval['rel2eval_idx'] = rel2eval_idx
        return tmp_m2eval

    def load_rel_mapping_eval(self, model_list, cache=False):
        tmp_m2eval = {m: {} for m in model_list} if not cache else self.context_resource
        for m in model_list:
            file_name = self.in_dir + f'rel_mapping_eval/{m}_rel_mapping_eval.pt'
            rel_eval = torch.load(file_name)
            tmp_m2eval[m].update({'rel_eval': rel_eval})
        rel2eval_idx = common_utils.load_json(self.in_dir + f"rel_mapping_eval/rel2eval_idx.json",
                                              object_hook=common_utils.jsonkeys2int)
        tmp_m2eval['rel2eval_idx'] = rel2eval_idx
        return tmp_m2eval

    def load_ent_degree_eval(self, model_list, cache=False):
        targets = ['in', 'out', 'entity']
        tmp_m2eval = {m: {} for m in model_list} if not cache else self.context_resource
        for m in model_list:
            for t in targets:
                file_name = self.in_dir + f'degree_eval/{m}_{t}_degree_eval.pt'
                t_degree_eval = torch.load(file_name)
                tmp_m2eval[m].update({f'{t}_degree_eval': t_degree_eval})
        for t in targets:
            ent2eval_idx = common_utils.load_json(self.in_dir + f"degree_eval/{t}_id2eval_idx.json",
                                                  object_hook=common_utils.jsonkeys2int)
            tmp_m2eval[f'{t}_id2eval_idx'] = ent2eval_idx
        return tmp_m2eval

    def load_total_eval(self, model_list, cache=False):
        tmp_m2eval = {m: {} for m in model_list} if not cache else self.context_resource
        for m in model_list:
            total_eval = torch.load(self.in_dir + f"total_eval/{m}_eval.pt")
            tmp_m2eval[m].update({'total_eval': total_eval})
        return tmp_m2eval
