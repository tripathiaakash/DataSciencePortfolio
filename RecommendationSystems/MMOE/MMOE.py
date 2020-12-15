import torch
from torch import nn

class TaskTower(nn.Module):
    def __init__(self,
                 units,
                 **kwargs):
        super(TaskTower, self).__init__(**kwargs)
        self.units = units
        self.layer_1 = nn.Linear(self.units, 8)
        self.layer_1_activation = nn.ReLU()
        self.layer_2 = nn.Linear(8, 1)
        self.layer_2_activation = nn.Sigmoid()

    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_1_activation(out)
        out = self.layer_2(out)
        out = self.layer_2_activation(out)
        return out
      
      
class MMOE(nn.Module):
    def __init__(self,
                units,
                num_experts,
                num_tasks,
                input_dimension,
                use_expert_bias=True,
                use_gate_bias=True,
                **kwargs):

        super(MMOE, self).__init__(**kwargs)
        
        ### Network hyper parameters
        self.units = units
        self.num_experts = num_experts
        self.num_tasks= num_tasks

        self.input_dimension = input_dimension

        ## Weight parameters
        self.expert_kernels = None
        self.gate_kernels = None

        ## Activation Parameters
        self.expert_activation = nn.ReLU()
        self.gate_activation = nn.Softmax(dim=1)

        ## Bias parameter
        self.expert_bias = None
        self.gate_bias = None
        self.use_expert_bias=use_expert_bias
        self.use_gate_bias=use_gate_bias

        self.expert_kernels = torch.empty(self.input_dimension, self.units, self.num_experts)
        self.expert_kernels = nn.init.xavier_uniform_(self.expert_kernels)
        self.expert_kernels = nn.parameter.Parameter(self.expert_kernels, requires_grad=True)

        if self.use_expert_bias:
            self.expert_bias = torch.empty(self.units. self.num_experts)
            self.expert_bias = nn.init.zeros_(self.expert_bias)
            self.expert_bias = nn.parameter.Parameter(self.expert_bias, requires_grad=True)

        self.gate_kernels = []
        
        ### Assuming NUmber of Tasks are 2 for now

        self.gate_kernels.append(nn.Linear(self.input_dimension, self.num_experts, bias=self.use_gate_bias))
        self.gate_kernels.append(nn.Linear(self.input_dimension, self.num_experts, bias=self.use_gate_bias))

        self.gate_kernels = nn.ModuleList(self.gate_kernels)

        self.towers = []
        self.tower1 = TaskTower(self.units)
        self.tower2 = TaskTower(self.units)
        self.towers.append(self.tower1)
        self.towers.append(self.tower2)

    def forward(self, x):
        final_outputs = []

        expert_outputs = torch.tensordot(a=x, b=self.expert_kernels, dims = 1)
        if self.use_expert_bias:
            expert_outputs = expert_outputs + self.expert_bias
        expert_outputs = self.expert_activation(expert_outputs)

        for index, gate_kernel in enumerate(self.gate_kernels):
            gate = gate_kernel(x)
            gate = self.gate_activation(gate)
            gate = gate.unsqueeze(2)
            gated_output = torch.bmm(expert_outputs, gate)
            gated_output = gated_output.squeeze(2)

            tower = self.towers[index]
            output = tower(gated_output)
            final_outputs.append(output)

        return final_outputs
