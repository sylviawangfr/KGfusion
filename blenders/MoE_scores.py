from pykeen.evaluation import RankBasedEvaluator
from pykeen.typing import LABEL_HEAD, LABEL_TAIL
from torch.utils.data import DataLoader
from tqdm import tqdm
from blenders.blender_utils import eval_with_blender_scores, Blender
from common_utils import format_result, save_to_file
from context_load_and_run import load_score_context
from features.feature_scores_only_dataset import ScoresOnlyDataset
from lp_kge.lp_pykeen import get_all_pos_triples
import torch
import torch.nn.functional as F
import argparse
from tutel import system, moe
from tutel import net


if torch.cuda.is_available():
    dist = system.init_data_model_parallel(backend='nccl')
else:
    dist = system.init_data_model_parallel(backend='gloo')


class CustomGate(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        torch.manual_seed(1)
        self.register_parameter(name='wg', param=torch.nn.Parameter(torch.randn([params.model_dim,
                                                                                 params.num_global_experts]) * 1e-3))

    def forward(self, x):
        return torch.matmul(x, self.wg)


class CustomExpert(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        torch.manual_seed(dist.global_rank + 1)
        self.register_parameter(name='batched_fc1_w', param=torch.nn.Parameter(torch.randn([params.num_local_experts, params.model_dim, params.hidden_size]) * 1e-3))
        self.register_parameter(name='batched_fc2_w', param=torch.nn.Parameter(torch.randn([params.num_local_experts, params.hidden_size, params.model_dim]) * 1e-3))
        self.register_parameter(name='batched_fc1_bias', param=torch.nn.Parameter(torch.zeros([params.num_local_experts, 1, params.hidden_size])))
        self.register_parameter(name='batched_fc2_bias', param=torch.nn.Parameter(torch.zeros([params.num_local_experts, 1, params.model_dim])))
        for x in self.parameters(): setattr(x, 'skip_allreduce', True)

    def forward(self, x):
        y = torch.add(torch.matmul(x, self.batched_fc1_w), self.batched_fc1_bias)
        y = F.relu(y)
        y = torch.add(torch.matmul(y, self.batched_fc2_w), self.batched_fc2_bias)
        return y


class CustomMoE(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.gate = CustomGate(params)
        self.expert = CustomExpert(params)

    def forward(self, x, k=2):
        logits = self.gate(x)
        scores = F.softmax(logits, dim=-1)
        crit, l_aux = moe.top_k_routing(scores, top_k=k)
        y = moe.fast_encode(x, crit)
        y = net.all_to_all(y, 1, 0)
        y = self.expert(y)
        y = net.all_to_all(y, 0, 1)
        output = moe.fast_decode(y, crit)
        return output, l_aux


def train_MoE(x, y, params):
    model = CustomMoE(params).to(dist.local_device)
    torch.manual_seed(dist.global_rank + 1)
    # data = torch.randn([128, params.model_dim], device=dist.local_device)
    # label = torch.LongTensor(128).random_(2).to(dist.local_device)
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-1)

    for i in range(params.epoch):
        t_start = system.record_time()
        optimizer.zero_grad()
        result, l_aux = model(x)
        result = F.log_softmax(result, dim=1)
        loss = F.nll_loss(result, y) + 0.0001 * l_aux
        loss.backward()

        for p in model.parameters():
            if not hasattr(p, 'skip_allreduce'):
                p.grad = net.simple_all_reduce(p.grad)
        optimizer.step()
        t_stop = system.record_time()
        dist.dist_print('STEP-%d: loss = %.5f, step_time = %.3f s' % (i, loss, t_stop - t_start))
    return model


def pred_with_MoE(pred_features, moe_layer):
    # moe_layer = moe_layer.to(dist.local_device)
    # In distributed model, you need further skip doing allreduce on global parameters that have `skip_allreduce` mask,
    # e.g.
    #    for p in moe_layer.parameters():
    #        if hasattr(p, 'skip_allreduce'):
    #            continue
    #        dist.all_reduce(p.grad)
    # Forward MoE:
    test_dataloader = DataLoader(pred_features.numpy(), batch_size=1000)
    individual_scores = []
    for batch in tqdm(test_dataloader):
        batch = batch.to(dist.local_device)
        weighted_scores, _ = moe_layer(batch)
        batch_scores = F.softmax(weighted_scores, dim=1)
        individual_scores.extend(batch_scores[:, 1])
    h_preds, t_preds = torch.chunk(torch.as_tensor(individual_scores), 2, 0)
    return h_preds, t_preds


def get_moe_params(params):
    model_list = params.models
    num_model = len(model_list)
    params.model_dim = num_model
    params.num_local_experts = 2
    params.num_global_experts = params.num_local_experts * dist.global_size
    params.hidden_size = num_model * 2
    params.epoch = 500
    # params.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return params


class MOEBlender(Blender):
    def __init__(self, params):
        super().__init__(params)
        self.context = load_score_context(params.models,
                                          in_dir=params.work_dir,
                                          calibration=True
                                          )

    def aggregate_scores(self):
        # read data
        all_pos_triples = get_all_pos_triples(self.dataset)
        work_dir = self.params.work_dir
        models_context = self.context
        dev_feature_dataset = ScoresOnlyDataset(self.dataset.validation.mapped_triples,
                                                models_context,
                                                all_pos_triples,
                                                num_neg=self.params.num_neg)
        test_feature_dataset = ScoresOnlyDataset(self.dataset.testing.mapped_triples, models_context, all_pos_triples)
        pos, neg = dev_feature_dataset.get_all_dev_examples()
        inputs = torch.cat([pos, neg], 0)
        labels = torch.cat([torch.ones(pos.shape[0]),
                            torch.zeros(neg.shape[0])], 0)
        labels = labels.type(torch.LongTensor)
        pred_features = test_feature_dataset.get_all_test_examples()

        # train MoE
        moe_params = get_moe_params(params=self.params)
        moe_layer = train_MoE(inputs, labels, moe_params)
        # predict
        h_preds, t_preds = pred_with_MoE(pred_features, moe_layer)
        # restore format that required by pykeen evaluator
        ht_scores = [h_preds, t_preds]
        evaluator = RankBasedEvaluator()
        relation_filter = None
        for ind, target in enumerate([LABEL_HEAD, LABEL_TAIL]):
            relation_filter = eval_with_blender_scores(
                batch=test_feature_dataset.mapped_triples,
                scores=ht_scores[ind],
                target=target,
                evaluator=evaluator,
                all_pos_triples=all_pos_triples,
                relation_filter=relation_filter,
            )
        result = evaluator.finalize()
        str_re = format_result(result)
        option_str = f"{self.params.dataset}_{'_'.join(self.params.models)}_MoE"
        save_to_file(str_re, work_dir + f"{option_str}.log")
        print(f"{option_str}:\n{str_re}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment settings")
    parser.add_argument('--models', type=str, default="TuckER_RotatE")
    parser.add_argument('--dataset', type=str, default="UMLS")
    parser.add_argument('--work_dir', type=str, default="../outputs/umls/")
    parser.add_argument('--num_neg', type=int, default=4)
    args = parser.parse_args()
    args.models = args.models.split('_')
    wab = MOEBlender(args)
    wab.aggregate_scores()
