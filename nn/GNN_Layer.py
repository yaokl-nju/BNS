import torch
from torch.nn import Parameter
from utils.init_func import softmax
from torch_sparse import SparseTensor
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, zeros

class GCNConv(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 normalize: bool = False,
                 ):
        super(GCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.reset_parameters()

    def forward(self, x, graph=None):
        output = self.lin(graph @ x)
        if self.normalize:
            output = F.normalize(output, p=2., dim=-1)
        return output

    def reset_parameters(self):
        glorot(self.lin.weight)
        if self.lin.bias is not None:
            zeros(self.lin.bias)

class SAGEConv(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 concat=False,
                 normalize=False,
                 ):
        super(SAGEConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.concat = concat
        self.normalize = normalize

        bias_l = bias
        self.lin_l = torch.nn.Linear(in_channels, out_channels, bias=bias_l)
        self.lin_r = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        if self.lin_l.bias is not None:
            zeros(self.lin_l.bias)
        if self.lin_r.bias is not None:
            zeros(self.lin_r.bias)

    def forward(self, x, graph=None):
        out_l = self.lin_l(graph @ x)
        out_r = self.lin_r(x[:out_l.size(0)])
        output = torch.cat([out_l, out_r], dim=1) if self.concat else out_l + out_r
        if self.normalize:
            output = F.normalize(output, p=2., dim=-1)
        return output

class GATConv(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 n_heads=1,
                 bias = True,
                 concat=True,
                 ):
        super(GATConv, self).__init__()
        self.n_heads = n_heads
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.concat=concat

        self.lin_l = torch.nn.Linear(self.in_dim, self.out_dim * self.n_heads, bias=False)
        self.lin_r = torch.nn.Linear(self.in_dim, self.out_dim * self.n_heads if concat else self.out_dim, bias=False)

        self.a1 = torch.nn.Linear(self.in_dim, self.n_heads, bias=False)
        self.a2 = torch.nn.Linear(self.in_dim, self.n_heads, bias=False)

        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2)

        out_dim = self.out_dim * self.n_heads if concat else self.out_dim
        self.bias = Parameter(torch.Tensor(out_dim)) if bias is not None else None
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.a1.weight)
        glorot(self.a2.weight)
        if self.bias is not None:
            zeros(self.bias)

    def forward(self, x, graph=None):
        num_nodes = graph[0].max() + 1
        x_l = self.lin_l(x).view(-1, self.n_heads, self.out_dim)
        x_r = self.lin_r(x[:num_nodes])

        attn1 = self.a1(x).view(-1, self.n_heads)
        attn2 = self.a2(x).view(-1, self.n_heads)

        attn = attn1[graph[0]] + attn2[graph[1]]
        attn = self.leaky_relu(attn)
        attn_d = softmax(attn, graph[0], num_nodes=x.size(0))

        out = []
        for i in range(self.n_heads):
            attn_adj = SparseTensor(row=graph[0], col=graph[1], value=attn_d[:, i], sparse_sizes=(num_nodes, x.size(0)))
            out.append((attn_adj @ x_l[:, i, :]).unsqueeze(1))
        out = torch.cat(out, dim=1)

        ### w/o gate
        out = out.flatten(1, -1) if self.concat else out.sum(1)
        out += x_r

        if self.bias is not None:
            out = out + self.bias
        return out


from itertools import product
from torch import Tensor
class PNAConv(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 concat=False,
                 normalize=False,
                 aggregators = None,
                 scalers = None,
                 ):
        super(PNAConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.concat = concat
        self.normalize = normalize
        self.aggregators = aggregators
        self.scalers = scalers

        n_heads = len(scalers) * len(aggregators)
        self.pre_lins = torch.nn.ModuleList([
            torch.nn.Linear(in_channels, out_channels, bias=False)
            for _ in range(len(aggregators) * len(scalers))
        ])
        self.lin = torch.nn.Linear(in_channels, out_channels * n_heads if concat else out_channels, bias=False)
        self.bias = Parameter(torch.Tensor(out_channels * n_heads if concat else out_channels)) if bias is not None else None
        self.reset_parameters()

        self.EPS = 1e-7

    def reset_parameters(self):
        for lin in self.pre_lins:
            glorot(lin.weight)
        glorot(self.lin.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, adj_t, avg_deg=None):
        out = self.message_and_aggregate(adj_t, x, avg_deg)
        return out

    def message_and_aggregate(self, adj_t, x, avg_deg):
        deg = adj_t.storage.rowcount().to(x.dtype).view(-1, 1)
        ### new version
        x_g = {}
        for aggr in self.aggregators:
            if aggr == 'mean':
                x_g['mean'] = adj_t @ x
            if aggr == 'std':
                x_g['std'] = ((adj_t @ (x ** 2) - (adj_t @ x) ** 2).relu_() + self.EPS).sqrt()
        out = []
        for (aggr, scaler), pre_lin in zip(
                product(self.aggregators, self.scalers), self.pre_lins):
            h = pre_lin(x_g[aggr])
            if scaler == 'amplification':
                h *= (deg + 1).log() / avg_deg['log']
            elif scaler == 'attenuation':
                h *= avg_deg['log'] / ((deg + 1).log() + self.EPS)
            out.append(h.unsqueeze(1))
        out = torch.cat(out, dim=1)

        ### w/o gate
        out = out.flatten(1, -1) if self.concat else out.sum(1)
        out += self.lin(x[:out.size(0)])

        if self.bias is not None:
            out = out + self.bias
        return out