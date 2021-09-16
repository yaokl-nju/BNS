import torch
import pandas as pd

def subgraph(ids_0, dataset):
    num_nodes = dataset.num_nodes
    idx = dataset.neigh_index(ids_0)
    rows, cols = dataset.data.edge_index[:, idx]
    weights = dataset.data.edge_weight[idx]

    reindex = torch.zeros(num_nodes, dtype=torch.long) - 1
    reindex[ids_0] = torch.arange(ids_0.size(0))

    rows = reindex[rows]

    if dataset.args.dataset != 'ogbn-papers100M':
        record = torch.zeros(num_nodes, dtype=torch.bool)
        record[ids_0] = True
        record[cols] = True
        ids_map = torch.where(record)[0]
        del record
    else:
        ids_map = torch.LongTensor(pd.unique(torch.cat([ids_0, cols]).numpy()))
    ids_col = ids_map[reindex[ids_map] < 0]
    ids_map = torch.cat([ids_0, ids_col])
    reindex[ids_col] = torch.arange(ids_col.size(0)) + ids_0.size(0)
    edge_index = torch.stack([rows, reindex[cols]], dim=0)
    graph = torch.sparse.FloatTensor(edge_index, weights.view(-1), (ids_0.size(0), ids_map.size(0)))
    return graph, ids_map
