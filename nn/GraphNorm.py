import torch.nn as nn
from torch.nn import Parameter
import torch
from torch.nn import init

class GraphNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1):
        super(GraphNorm, self).__init__()
        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))
        self.momentum = momentum
        self.eps = 1e-5
        self.meanscale = Parameter(torch.ones(num_features))

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        self.reset_parameters()

    def forward(self, input):
        (n, _) = input.size()
        if self.training:
            mean = input.mean(0)
            std2 = ((input - self.meanscale * mean) ** 2).mean(0)
            # std2 = input.var(0, unbiased=False)
            with torch.no_grad():
                self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * mean.data
                self.running_var = (1.0 - self.momentum) * self.running_var + (self.momentum * (n / (n - 1.0))) * std2.data
        else:
            mean = self.running_mean
            std2 = self.running_var

        out = (input - self.meanscale * mean) * (1.0 / (std2 + self.eps).sqrt())
        out = out * self.weight + self.bias
        return out

    def reset_parameters(self) -> None:
        init.ones_(self.weight)
        init.zeros_(self.bias)
        init.zeros_(self.running_mean)
        init.ones_(self.running_var)
        init.ones_(self.meanscale)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(GraphNorm, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

