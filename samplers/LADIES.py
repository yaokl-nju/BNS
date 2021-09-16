import torch
import numpy as np
from torch_scatter import scatter_add
import pandas as pd

def LADIES(ids_0, dataset):
    '''
    :param n_n_l: number of neighbor per layer
    '''
    num_nodes = dataset.data.x.size(0)
    ids_cur = ids_0
    ids_record = torch.zeros(num_nodes, dtype=torch.bool)
    ids_record[ids_cur] = True
    edges_row, edges_col, edges_weight = [], [], []

    n_n_l = int(dataset.args.k * dataset.args.bsize)
    for i in range(dataset.args.layer_num):
        '''
            1. get all neighbors of ids_cur, cols_i
            2. cal probs for cols_i
            3. sample from cols_i
            4. get all edges between ids_cur and cur_neigh_sample
            5. refine edge_weights of edges 
        '''
        if dataset.args.dataset == 'ogbn-proteins':
            subgraph = dataset.graph_sp[ids_cur]
            rows, cols, weights = subgraph.storage.row(), subgraph.storage.col(), subgraph.storage.value()
            subdeg = subgraph.storage.rowcount()
            rows = torch.repeat_interleave(ids_cur, subdeg)
        else:
            neigh_idx = dataset.neigh_index(ids_cur)
            rows, cols = dataset.data.edge_index[:, neigh_idx]
            weights = dataset.data.edge_weight[neigh_idx].view(-1)
        if ids_cur.size(0) < num_nodes:
            probs = scatter_add(weights ** 2, cols, dim=0, dim_size=num_nodes)
            probs /= probs.sum()
            ind = torch.where(probs)[0].numpy()
            probs_i = probs[ind]
        else:
            probs = dataset.node_probs
            probs_i = probs
            ind = num_nodes
        if probs_i.size(0) > n_n_l:
            cols_sample = np.random.choice(ind, n_n_l, p=probs_i.numpy(), replace=False)
        else:
            cols_sample = ind if isinstance(ind, np.ndarray) else torch.arange(ind)
        nodes_bool = torch.zeros(num_nodes, dtype=torch.bool)
        nodes_bool[cols_sample] = True

        sample_index = torch.where(nodes_bool[cols])[0]
        rowids = rows[sample_index]
        colids = cols[sample_index]
        weights = weights[sample_index]

        if dataset.args.dataset != 'ogbn-papers100M':
            ids_cur = torch.where(nodes_bool)[0]
        else:
            ids_cur = torch.LongTensor(pd.unique(cols_sample))

        edges_row.append(rowids)
        edges_col.append(colids)
        edge_weight_i = weights * (1.0 / (probs[edges_col[-1]] + 1e-8))
        edges_weight.append(dataset.row_normalize(edge_weight_i, edges_row[-1], num_nodes).view(-1, 1))
        # edges_weight = [dataset.row_normalize(
        #     torch.ones(edges_col[-1].size(0)), edges_row[-1], num_nodes).view(-1, 1)] + edges_weight
        ids_record[cols_sample] = True

    ### reindex ids in top-down manner
    ########################################
    ids_all = torch.where(ids_record)[0]
    remap = torch.zeros(num_nodes, dtype=torch.long)
    remap[ids_all] = torch.arange(ids_all.size(0))
    edges_col_new, edges_row_new, ids_0_re = [], [], remap[ids_0]
    for i in range(dataset.args.layer_num):
        edges_row_new.append(remap[edges_row[i]])
        edges_col_new.append(remap[edges_col[i]])

    reindex = torch.zeros(ids_all.size(0), dtype=torch.long) - 1
    reindex[ids_0_re] = torch.arange(ids_0_re.size(0))
    edge_row_i, oid_list, graph = reindex[edges_row_new[0]], [ids_0], []
    offset, oid_all = ids_0_re.size(0), ids_0_re
    for i in range(dataset.args.layer_num):
        oid = relabel_cpu(reindex, edges_col_new[i], oid_all)
        reindex[oid] = offset + torch.arange(oid.size(0))
        offset += oid.size(0)
        oid_all = torch.cat([oid_all, oid])
        edge_index_i = torch.stack([edge_row_i, reindex[edges_col_new[i]]], dim=0)
        if dataset.args.model != 'GAT':
            graph.append(torch.sparse.FloatTensor(edge_index_i, edges_weight[i].view(-1),
                                                  (oid_list[-1].size(0), offset)))
        else:
            graph.append(edge_index_i)
        if i < dataset.args.layer_num - 1:
            edge_row_i = reindex[edges_row_new[i + 1]]
        oid_list.append(ids_all[oid_all])
    idx_re = reindex[remap[ids_0]]
    graph.reverse()
    oid_list.reverse()
    ########################################
    return [graph, oid_list[0], ids_0, idx_re]

def relabel_cpu(reindex, edges_col, oid_all):
    bool_id = torch.zeros(reindex.size(0), dtype=torch.bool)
    bool_id[edges_col] = True
    bool_id[oid_all] = True
    oid = torch.where(bool_id)[0]
    nid = reindex[oid]
    return oid[nid < 0]