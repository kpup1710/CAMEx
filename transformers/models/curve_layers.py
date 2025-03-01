import math
import torch
from torch import nn
from torch.distributions.normal import Normal
import pdb 

softplus = nn.Softplus()
softmax = nn.Softmax(1)

class Conv1D_MEO(nn.Module):

    def __init__(self, input_size, output_size, layer_idx, config, noisy_gating=True, reduce_factor=16, bias=True):
        super(Conv1D_MEO, self).__init__()
        
        self.noisy_gating, self.moe_level = noisy_gating, config.moe_level
        self.n_experts, self.k = config.n_experts, config.k
        self.mean = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.std = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        input_dim = config.description_size // reduce_factor if self.moe_level == 'task' else input_size
        self.w_gate = nn.Parameter(torch.empty(input_dim, config.n_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.empty(input_dim, config.n_experts), requires_grad=True)
        
        # print('bias: ', bias)
        # instantiate experts
        self.weight = nn.Parameter(torch.empty(
            config.n_experts, output_size, input_size), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(
            config.n_experts, output_size), requires_grad=True) if bias else None
        self.res_weight = nn.Parameter(torch.empty(output_size, input_size), requires_grad=True)
        self.res_bias = nn.Parameter(torch.zeros(1,output_size), requires_grad=True) if bias else None
        nn.init.normal_(self.weight, std=0.02)
        nn.init.normal_(self.res_weight, std=0.02)
        nn.init.normal_(self.w_gate, std=0.02)
        nn.init.normal_(self.w_noise, std=0.02)
        # if self.moe_level == 'token':
        #     self.tokenatt = TokenAtt(input_size)
        if self.moe_level == 'task':
            self.task_proj = nn.Linear(config.description_size, config.description_size // reduce_factor)  # todo randomly init
        assert (self.k <= config.n_experts)
        self.output_size = output_size
        self.input_size = input_size
        self.T = 256
        self.ties_merging = True
        
        self.dim1, self.dim2 = self.get_nearest_factors_input()
        self.dim_out1, self.dim_out2 = self.get_nearest_factors_output()
        self.curve1_in  = nn.Parameter(torch.stack([torch.diag(torch.ones(self.dim1)) for _ in range(config.n_experts)], dim=0), requires_grad=True)
        self.curve2_in  = nn.Parameter(torch.stack([torch.diag(torch.ones(self.dim2)) for _ in range(config.n_experts)], dim=0), requires_grad=True)
        self.curve1_out = nn.Parameter(torch.stack([torch.diag(torch.ones(self.dim_out1)) for _ in range(config.n_experts)], dim=0), requires_grad=True)
        self.curve2_out = nn.Parameter(torch.stack([torch.diag(torch.ones(self.dim_out2)) for _ in range(config.n_experts)], dim=0), requires_grad=True)
        
        self.curve1_bias = nn.Parameter(torch.stack([torch.diag(torch.ones(self.dim_out1)) for _ in range(config.n_experts)], dim=0), requires_grad=True) if bias else None
        self.curve2_bias = nn.Parameter(torch.stack([torch.diag(torch.ones(self.dim_out2)) for _ in range(config.n_experts)], dim=0), requires_grad=True) if bias else None
        
        
    def get_nearest_factors_input(self):
        sqrt_inp_size = int(math.sqrt(self.input_size))
        while self.input_size % sqrt_inp_size != 0:
            sqrt_inp_size = sqrt_inp_size - 1
        res = max(sqrt_inp_size, 1)
        return res, self.input_size // res
    
    def get_nearest_factors_output(self):
        sqrt_inp_size = int(math.sqrt(self.output_size))
        while self.output_size % sqrt_inp_size != 0:
            sqrt_inp_size = sqrt_inp_size - 1
        res = max(sqrt_inp_size, 1)
        return res, self.output_size // res

        
    
    def init_res(self):
        with torch.no_grad():
            self.res_weight.data = self.weight[0].data.clone()
            self.res_bias.data = self.bias[0].data.clone()
            # self.weight.data = self.weight.data 
        # self.res_weight.requires_grad = False
        # self.res_bias.requires_grad = False
                    
    def forward(self, x, task_embeddings=None, loss_coef=1e-5):
        loss = None
        batch_size = x.shape[0]
        L = x.shape[1]
        d = x.shape[-1]
        
        N = L // self.T
        
        if self.moe_level == 'task' and task_embeddings is not None:
            task_embeddings = self.task_proj(task_embeddings)
            task_embeddings = task_embeddings.mean(1)
            gates, load = noisy_top_k_gating(self, task_embeddings, self.training)
        else:
            x = x.view(batch_size*N, self.T, d)
            if self.ties_merging:
                self.k = self.n_experts
            # self.k = int(torch.randint(4,8,(1,)))
            gates, load = noisy_top_k_gating(self, x.mean(1), self.training)
        # calculate importance loss
        importance = gates.view(-1, self.n_experts).sum(0)
        loss = cv_squared(importance) + cv_squared(load)
        loss *= loss_coef
        
        e_first = gates.view(batch_size, N, self.n_experts)[:, 0].clone()
        e_first = e_first.detach()
        gates = torch.roll(gates, 1, 0).view(batch_size, N, self.n_experts)
        gates[:, 0] = e_first
        gates = gates.view(batch_size*N, self.n_experts)
        
        # gates = torch.cat([torch.ones([batch_size*N, 1]), gates], dim=1)
        
        # if not self.ties_merging:
        #     if self.training:
        #         m = torch.ones([batch_size*N, self.n_experts, self.output_size, self.input_size + 1], dtype=torch.bool, device=gates.device)*0.9
     
        #     else:
        #         m = torch.ones([batch_size*N, self.n_experts, self.output_size, self.input_size + 1], dtype=torch.bool, device=gates.device)*0.9
        #     masks = torch.bernoulli(m).to(torch.bool)
        #     mask_w = masks[:,:,:,:-1]
        #     mask_b = masks[:,:,:,-1]
        # else:
        #     mask_w = torch.sign(self.weight - self.res_weight).to(torch.int16)
        #     mask_w *= (mask_w == torch.sign(torch.sum(self.weight - self.res_weight, dim=0))).to(torch.int16)
        #     mask_b = torch.sign(self.bias - self.res_bias).to(torch.int16)
        #     mask_b *= (mask_b == torch.sign(torch.sum(self.bias - self.res_bias, dim=0))).to(torch.int16)
        #     mask_w *= torch.bernoulli(torch.ones(self.res_weight.shape,dtype=torch.bool,device=gates.device)*0.9).to(bool)
        #     mask_b *= torch.bernoulli(torch.ones(self.res_bias.shape,dtype=torch.bool,device=gates.device)*0.9).to(bool)
            
        
        if self.k * 2 < self.n_experts:
            index = torch.nonzero(gates)[:, -1:].flatten()
            gates = gates[gates != 0]                                                  
            expert_weights = torch.sum((gates.view(-1, 1, 1) * torch.index_select(self.weight, 0, index)).view(batch_size*N, self.k, self.weight.size()[-2], self.weight.size()[-1]), dim=1)
        else:
            # res_task_weights = torch.randn(self.weight.shape)
            res_task_weights = self.weight - self.res_weight # [n, c_in, c_out]
            res_task_weights = res_task_weights.view(-1, self.dim_out1, self.dim_out2, self.dim1, self.dim2)
            
            res_task_weights = torch.einsum("bij, bjklm->biklm", self.curve1_out, res_task_weights)
            res_task_weights = torch.einsum("bik, bjklm->bjilm", self.curve2_out, res_task_weights)
            res_task_weights = torch.einsum("bil, bjklm->bjkim", self.curve1_in, res_task_weights)
            res_task_weights = torch.einsum("bim, bjklm->bjkli", self.curve2_in, res_task_weights)

            res_task_weights = res_task_weights.reshape(-1, self.output_size, self.input_size) 
            # expert_weights = torch.cat([self.res_weight.unsqueeze(0), self.weight], dim=0)
            expert_weights = self.res_weight + torch.sum(res_task_weights* gates.view(batch_size*N, -1, 1, 1), dim=1) 
            # expert_weights = torch.sum(torch.mul(self.weight, gates.view(batch_size*N, -1, 1, 1)), dim=1)
        # if self.moe_level == 'token':
        #     x = x + self.tokenatt(x)
        y = torch.einsum('bij, bkj->bik', x, expert_weights) 
        if self.bias is not None:
            if self.k * 2 < self.n_experts:
                expert_bias = torch.sum((gates.view(-1,1) * torch.index_select(self.bias, 0, index)).view(batch_size*N, self.k, self.bias.size()[-1]), dim=1)
            else:
                # res_task_bias = torch.randn(self.bias.shape)
                res_task_bias = self.bias - self.res_bias
                res_task_bias = res_task_bias.view(-1, self.dim_out1, self.dim_out2)
                
                
                res_task_bias = torch.einsum("bki, bij->bkj", self.curve1_bias, res_task_bias)
                res_task_bias = torch.einsum("bkj, bij->bik", self.curve2_bias, res_task_bias)
                res_task_bias = res_task_bias.reshape(-1, self.output_size)
                # expert_bias = torch.cat([self.res_bias, self.bias], dim=0)
                expert_bias = self.res_bias + torch.sum(torch.mul(res_task_bias, gates.view(batch_size*N, -1, 1)), dim=1) 
            y = y + expert_bias.unsqueeze(1)
        y = y.view(batch_size, L, self.output_size)
        return y, loss
    
def noisy_top_k_gating(layer, x, train, noise_epsilon=1e-2):
    clean_logits = x @ layer.w_gate
    if layer.noisy_gating and train:
        raw_noise_stddev = x @ layer.w_noise
        noise_stddev = ((softplus(raw_noise_stddev) + noise_epsilon))
        noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
        logits = noisy_logits
    else:
        logits = clean_logits
    # calculate topk + 1 that will be needed for the noisy gates
    top_logits, top_indices = logits.topk(min(layer.k + 1, layer.n_experts), dim=-1)
    top_k_logits = top_logits[:, : layer.k]
    top_k_indices = top_indices[:, : layer.k]
    top_k_gates = softmax(top_k_logits)
    zeros = torch.zeros_like(logits, requires_grad=True, dtype=top_k_gates.dtype).to(x.device)
    gates = zeros.scatter(1, top_k_indices, top_k_gates).to(x.device)
    if layer.noisy_gating and layer.k < layer.n_experts and train:
        load = (_prob_in_top_k(layer, clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
    else:
        load = _gates_to_load(gates)
    return gates, load

def _prob_in_top_k(layer, clean_values, noisy_values, noise_stddev, noisy_top_values):
    batch = clean_values.size(0)
    m = noisy_top_values.size(1)
    top_values_flat = noisy_top_values.flatten()
    normal = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    threshold_positions_if_in = torch.arange(batch).to(clean_values.device) * m + layer.k
    threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
    is_in = torch.gt(noisy_values, threshold_if_in)
    threshold_positions_if_out = threshold_positions_if_in - 1
    threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
    # is each value currently in the top k.
    prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
    prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
    prob = torch.where(is_in, prob_if_in, prob_if_out)
    return prob

def cv_squared(x):
    eps = 1e-10
    if x.shape[0] == 1:
        return torch.Tensor([0]).to(x.device)
    return x.float().var() / (x.float().mean() ** 2 + eps)

def _gates_to_load(gates):
    return (gates > 0).sum(0)

class SparseDispatcher(object):

    def __init__(self, n_experts, gates):

        self._gates = gates
        self._n_experts = n_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        # assigns samples to experts whose gate is nonzero
        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            _nonzero_gates = self._nonzero_gates.unsqueeze(-1) if stitched.dim() == 3 else self._nonzero_gates
            stitched = stitched.mul(_nonzero_gates)
        if stitched.dim() == 2:
            zeros = torch.zeros(self._gates.size()[0], expert_out[-1].size(1), requires_grad=True).to(stitched.device)
        else:
            zeros = torch.zeros(self._gates.size()[0], expert_out[-1].size(-2), expert_out[-1].size(-1),
                                requires_grad=True).to(stitched.device)

        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float()).to(stitched.device)
        # add eps to all zero values in order to avoid nans when going back to log space
        # combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined

    def expert_to_gates(self):
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class SingleExpert(nn.Module):

    def __init__(self, input_size, hidden_size, activation=None, bias=False):
        super().__init__()
        self.dense = nn.Linear(input_size, hidden_size, bias=bias)
        self.activation = activation

    def forward(self, hidden_states):
        output = self.dense(hidden_states)
        if self.activation is not None:
            output = self.activation(output)
        return output


class MoE(nn.Module):

    def __init__(self, input_size, hidden_size, config, noisy_gating=True, reduce_factor=16, bias=False):
        super(MoE, self).__init__()
        self.k = config.k
        self.noisy_gating, self.moe_level = noisy_gating, config.moe_level
        self.n_experts, self.k = config.n_experts, config.k
        self.input_size, self.output_size = input_size, hidden_size
        # instantiate experts
        input_dim = config.description_size // reduce_factor if self.moe_level == 'task' else input_size
        self.w_gate = nn.Parameter(torch.zeros(input_dim, config.n_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_dim, config.n_experts), requires_grad=True)
        self.experts = nn.ModuleList([SingleExpert(input_size, hidden_size, bias=bias) for i in range(config.n_experts)])
        self.mean = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.std = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        if self.moe_level == 'task':
            self.task_proj = nn.Linear(config.description_size,
                                       config.description_size // reduce_factor)  # todo randomly init
        assert (self.k <= config.n_experts)

    def forward(self, x, task_embeddings=None, loss_coef=1e-2):
        original_shape = list(x.shape[:-1])
        if self.moe_level == 'task' and task_embeddings is not None:
            task_embeddings = self.task_proj(task_embeddings)
            task_embeddings = task_embeddings.mean(1)
            gates, load = noisy_top_k_gating(self, task_embeddings, self.training)
        elif self.moe_level == 'sequence':
            gates, load = noisy_top_k_gating(self, x.mean(-2), self.training)
        else:
            x = x.reshape(-1, self.input_size)
            gates, load = noisy_top_k_gating(self, x, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        loss = cv_squared(importance) + cv_squared(load)
        loss *= loss_coef
        dispatcher = SparseDispatcher(self.n_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.n_experts)]
        y = dispatcher.combine(expert_outputs)
        y = y.reshape(original_shape + [self.output_size])
        return y, loss


class MEO(nn.Module):

    def __init__(self, input_size, output_size, config, noisy_gating=True, reduce_factor=64, rank = 1, bias=False):
        super(MEO, self).__init__()
        self.noisy_gating, self.moe_level = noisy_gating, config.moe_level
        self.n_experts, self.k = config.n_experts, config.k
        
        self.mean = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.std = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        input_dim = input_size
        self.w_gate = nn.Parameter(torch.empty(input_dim, config.n_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_dim, config.n_experts), requires_grad=True)
        self.res_weight = nn.Parameter(torch.Tensor(output_size, input_size), requires_grad=True)
        self.res_bias = nn.Parameter(torch.zeros(1,output_size), requires_grad=True) if bias else None
        # instantiate experts
        self.weight = nn.Parameter(torch.Tensor(config.n_experts, output_size, input_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(config.n_experts, output_size), requires_grad=True)
        else:
            self.bias = None

        assert (self.k <= self.n_experts)
        
        self.ties_merging = False
        self.input_size = input_size
        self.output_size = output_size
        self.dim1, self.dim2 = self.get_nearest_factors_input()
        self.dim_out1, self.dim_out2 = self.get_nearest_factors_output()
        self.curve1_in  = nn.Parameter(torch.stack([torch.diag(torch.ones(self.dim1)) for _ in range(config.n_experts)], dim=0), requires_grad=True)
        self.curve2_in  = nn.Parameter(torch.stack([torch.diag(torch.ones(self.dim2)) for _ in range(config.n_experts)], dim=0), requires_grad=True)
        self.curve1_out = nn.Parameter(torch.stack([torch.diag(torch.ones(self.dim_out1)) for _ in range(config.n_experts)], dim=0), requires_grad=True)
        self.curve2_out = nn.Parameter(torch.stack([torch.diag(torch.ones(self.dim_out2)) for _ in range(config.n_experts)], dim=0), requires_grad=True)
        
        self.drop = nn.Dropout2d(p=0.1)
        nn.init.normal_(self.weight, std=0.02)
        nn.init.normal_(self.res_weight, std=0.02)
        nn.init.normal_(self.w_gate, std=0.02)
        nn.init.normal_(self.w_noise, std=0.02)
        # self.curve1_bias = nn.Parameter(torch.stack([torch.diag(torch.ones(self.dim_out1)) for _ in range(config.n_experts)], dim=0), requires_grad=True) if bias else None
        # self.curve2_bias = nn.Parameter(torch.stack([torch.diag(torch.ones(self.dim_out2)) for _ in range(config.n_experts)], dim=0), requires_grad=True) if bias else None
        
    def get_nearest_factors_input(self):
        sqrt_inp_size = int(math.sqrt(self.input_size))
        while self.input_size % sqrt_inp_size != 0:
            sqrt_inp_size = sqrt_inp_size - 1
        res = max(sqrt_inp_size, 1)
        return res, self.input_size // res
    
    def get_nearest_factors_output(self):
        sqrt_inp_size = int(math.sqrt(self.output_size))
        while self.output_size % sqrt_inp_size != 0:
            sqrt_inp_size = sqrt_inp_size - 1
        res = max(sqrt_inp_size, 1)
        return res, self.output_size // res
    
    def init_res(self):
        with torch.no_grad():
            self.res_weight.data = self.weight[0].data.clone()
            # print('-------------------------------------------------------init-------------------------------------------')
            if self.bias:
                self.res_bias.data = self.bias[0].data.clone()
            # self.weight.data = self.weight.data 
        # self.res_weight.requires_grad = False
        # self.res_bias.requires_grad = False
        
    def forward(self, x, task_embeddings=None, loss_coef=1e-2):
        loss = None
        batch_size = x.shape[0]
        try:
            gates, load = noisy_top_k_gating(self, x.mean(-2), self.training)
        except:
            gates, load = noisy_top_k_gating(self, x.mean(-2), self.training)
        
        gates, load = noisy_top_k_gating(self, x.mean(-2), self.training)
        # calculate importance loss
        importance = gates.view(-1, self.n_experts).sum(0)
        loss = cv_squared(importance) + cv_squared(load)
        loss *= loss_coef
        
        if self.ties_merging:
            mask_w = torch.sign(self.weight - self.res_weight).to(torch.int8)
            mask_w *= (mask_w == torch.sign(torch.sum(self.weight - self.res_weight, dim=0))).to(torch.int8) 
            
        if self.k * 2 < self.n_experts:
            index = torch.nonzero(gates)[:, -1:].flatten()
            gates = gates[gates != 0]                                                  
            expert_weights = torch.sum((gates.view(-1, 1, 1) * torch.index_select(self.weight, 0, index)).view(batch_size, self.k, self.weight.size()[-2], self.weight.size()[-1]), dim=1)
        else:
            res_task_weights = self.weight - self.res_weight # [n, c_in, c_out]
            res_task_weights = res_task_weights.view(-1, self.dim_out1, self.dim_out2, self.dim1, self.dim2)
            
            res_task_weights = torch.einsum("bij, bjklm->biklm", self.curve1_out, res_task_weights)
            res_task_weights = torch.einsum("bik, bjklm->bjilm", self.curve2_out, res_task_weights)
            res_task_weights = torch.einsum("bil, bjklm->bjkim", self.curve1_in, res_task_weights)
            res_task_weights = torch.einsum("bim, bjklm->bjkli", self.curve2_in, res_task_weights)

            res_task_weights = res_task_weights.reshape(-1, self.output_size, self.input_size) 
            
            expert_weights = self.res_weight + 0.9*torch.sum(self.drop(torch.mul(res_task_weights, gates.view(batch_size, -1, 1, 1))), dim=1) 
            # expert_weights = self.res_weight + 0.5*torch.sum(torch.mul(res_task_weights, gates.view(batch_size, -1, 1, 1))*mask_w, dim=1) 
            # expert_weights = torch.sum(torch.mul(self.weight, gates.view(batch_size, -1, 1, 1)), dim=1)

        y = torch.einsum('bij,bkj->bik', x, expert_weights) 
        # pdb.set_trace()
        # try:
        #     print(loss)
        # except:
        #     pdb.set_trace()
        return y, loss