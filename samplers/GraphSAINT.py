import torch
from torch_sparse import SparseTensor
import pandas as pd

def GraphSAINT(ids_0, datasets):
    num_nodes = datasets.data.x.size(0)
    record_i = torch.zeros(num_nodes, dtype=torch.bool)
    record_i[ids_0] = True

    ids_cur = ids_0

    walk_length = datasets.args.walk_length
    for j in range(walk_length):
        rand_j = (torch.rand(ids_cur.size(0)) * datasets.deg[ids_cur]).to(torch.long)
        idx_j = rand_j + datasets.rowptr[ids_cur]
        ids_cur = datasets.data.edge_index[1, idx_j]
        record_i[ids_cur] = True

    ids_maps = torch.where(record_i)[0]
    idx_i = datasets.neigh_index(ids_maps)
    edge_index = datasets.data.edge_index[:, idx_i]
    idx_i_e = torch.where(record_i[edge_index[1]])[0]
    edge_index = edge_index[:, idx_i_e]
    ids_reindex = record_i.new_zeros(num_nodes, dtype=torch.long)
    ids_reindex[ids_maps] = torch.arange(0, ids_maps.size(0))
    edge_index = torch.stack([ids_reindex[edge_index[0]],ids_reindex[edge_index[1]]], dim=0)
    idx_re = ids_reindex[ids_0]

    # edges_weight_i = (self.data.edge_weight[idx_i[idx_i_e]].view(-1)
    #                   / self.alpha[idx_i[idx_i_e]])
    # edges_weight_i = self.row_normalize(edges_weight_i, edge_index_i[0], num_nodes).to(self.device)

    ### better results can be achieved via the following row-normalize
    edges_weight = datasets.row_normalize(torch.ones(edge_index.size(1)), edge_index[0], num_nodes)
    if datasets.args.model != 'GAT':
        ### slower.
        # graph = torch.sparse.FloatTensor(edge_index, edges_weight.view(-1), (ids_maps.size(0), ids_maps.size(0)))

        ### faster.
        graph = SparseTensor(row=edge_index[0], col=edge_index[1], value=edges_weight.view(-1),
                                  sparse_sizes=(ids_maps.size(0), ids_maps.size(0)))
    else:
        graph = edge_index
    return [graph, ids_maps, ids_0, idx_re]