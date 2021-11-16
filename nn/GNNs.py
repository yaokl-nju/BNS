from nn.GNN_Layer import *
import torch.nn.functional as F
import nn
from nn.GraphNorm import GraphNorm

class GNN(torch.nn.Module):
    def __init__(self, args):
        super(GNN, self).__init__()
        self.args = args
        self.act = torch.nn.ReLU()
        self.drop = torch.nn.ModuleList([])
        self.drop.append(torch.nn.Identity() if args.dataset == 'ogbn-proteins' else torch.nn.Dropout(args.drop))
        for i in range(args.layer_num - 2):
            self.drop.append(torch.nn.Dropout(args.drop) if args.drop != 0.0 else torch.nn.Identity())

        if args.model == 'SAGE-GCN':
            hidden_dim = args.hidden_dim
            in_dims = [args.num_features] + [args.hidden_dim] * (args.layer_num - 1)
            out_dim = hidden_dim if args.linproj else args.num_classes
            kwargs = [{}] * (args.layer_num)
            modelname = 'GCNConv'
        elif args.model == 'SAGE':
            hidden_dim = args.hidden_dim * 2 if args.concat else args.hidden_dim
            in_dims = [args.num_features] + [hidden_dim] * (args.layer_num - 1)
            out_dim = args.hidden_dim if args.linproj else args.num_classes
            kwargs = [{'concat': args.concat}] * (args.layer_num - 1) + \
                     [{'concat': args.concat if args.linproj else False}]
            modelname = 'SAGEConv'
        elif args.model == 'GAT':
            hidden_dim = args.hidden_dim * args.n_heads if args.concat else args.hidden_dim
            in_dims = [args.num_features] + [hidden_dim] * (args.layer_num - 1)
            out_dim = args.hidden_dim if args.linproj else args.num_classes
            kwargs = [{'n_heads': args.n_heads, 'concat': args.concat}] * (args.layer_num - 1) + \
                     [{'n_heads': args.n_heads, 'concat': args.concat if args.linproj else False}]
            modelname = 'GATConv'
        elif args.model == 'PNA':
            n_heads = len(args.aggregators) * len(args.scalers)
            hidden_dim = args.hidden_dim * n_heads if args.concat else args.hidden_dim
            in_dims = [args.num_features] + [hidden_dim] * (args.layer_num - 1)
            out_dim = args.hidden_dim if args.linproj else args.num_classes
            kwargs = [{'aggregators': args.aggregators, 'scalers': args.scalers, 'concat': args.concat}] * (args.layer_num - 1) + \
                     [{'aggregators': args.aggregators, 'scalers': args.scalers, 'concat': args.concat if args.linproj else False}]
            modelname = 'PNAConv'

        GNNConv = getattr(nn, modelname)
        self.convs = torch.nn.ModuleList([])
        for i in range(args.layer_num - 1):
            self.convs.append(GNNConv(in_dims[i], args.hidden_dim, **kwargs[i]))
        self.convs.append(GNNConv(in_dims[-1], out_dim, **kwargs[-1]))

        if args.linproj:
            self.drop.append(torch.nn.Dropout(args.drop))
            self.proj = torch.nn.Linear(in_dims[-1], args.num_classes)
            self.reset_parameters()
        else:
            self.drop.append(torch.nn.Identity())

        self.bns = torch.nn.ModuleList([])
        for i in range(args.layer_num - 1):
            self.bns.append(GraphNorm(in_dims[i + 1], momentum=0.9) if args.bn else torch.nn.Identity())
        if args.linproj:
            self.bns.append(GraphNorm(in_dims[-1], momentum=0.9) if args.bn else torch.nn.Identity())

        ### for PNA
        if self.args.model == 'PNA':
            deg = args.deg.to(torch.float)
            self.avg_deg = {
                'lin': deg.mean().item(),
                'log': (deg + 1).log().mean().item(),
            }

    def reset_parameters(self):
        glorot(self.proj.weight)
        if self.proj.bias is not None:
            zeros(self.proj.bias)

    def forward(self, feat, graph):
        kwargs = {'avg_deg': self.avg_deg} if self.args.model == 'PNA' else {}
        x = feat
        if not isinstance(graph, list):
            for i in range(self.args.layer_num - 1):
                x = self.convs[i](self.drop[i](x), graph)
                x = F.relu(self.bns[i](x), inplace=True)
            x = self.convs[-1](self.drop[-1](x), graph)
            if self.args.linproj:
                x = self.proj(F.relu(self.bns[-1](x), inplace=True))
        else:
            for i in range(self.args.layer_num - 1):
                x = self.convs[i](self.drop[i](x), graph[i], **kwargs)
                x = F.relu(self.bns[i](x), inplace=True)
            x = self.convs[-1](self.drop[-1](x), graph[-1], **kwargs)
            if self.args.linproj:
                x = self.proj(F.relu(self.bns[-1](x), inplace=True))
        probs = F.softmax(x, dim=-1)
        return probs, x

    def forward_layer(self, feat, graph, layer):
        if layer < self.args.layer_num - 1:
            x = self.convs[layer](self.drop[layer](feat), graph)
            x = F.relu(self.bns[layer](x), inplace=True)
        else:
            x = self.convs[-1](self.drop[-1](feat), graph)
            if self.args.linproj:
                x = self.proj(F.relu(self.bns[-1](x), inplace=True))
        return x
