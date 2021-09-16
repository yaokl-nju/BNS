import torch
from utils.init_func import choice_multi_range
import pandas as pd

def NS(ids_0, dataset):
    num_nodes = dataset.data.x.size(0)
    ids_cur = ids_0
    if dataset.args.dataset == 'ogbn-papers100M':
        ids_record = ids_0
    else:
        ids_record = torch.zeros(num_nodes, dtype=torch.bool)
        ids_record[ids_0] = True
    edges_row, edges_col, edges_weight = [], [], []

    s_n = dataset.args.s_n
    for i in range(dataset.args.layer_num):
        '''
            sample w/o replacement
        '''
        sample_mask = dataset.deg[ids_cur] > s_n
        ids_cur_1 = ids_cur[sample_mask]
        ids_cur_2 = ids_cur[torch.logical_not(sample_mask)]
        deg_i_1 = torch.zeros(ids_cur_1.size(0), dtype=torch.long) + s_n

        mask_1_1 = dataset.deg[ids_cur_1] < s_n * 10
        ids_cur_1_1, deg_i_1_1 = ids_cur_1[mask_1_1], deg_i_1[mask_1_1]
        idx_i_1_1 = choice_multi_range(dataset.deg[ids_cur_1_1].numpy(), deg_i_1_1.numpy())
        idx_i_1_1 = torch.LongTensor(idx_i_1_1) \
                  + torch.repeat_interleave(dataset.rowptr[ids_cur_1_1], deg_i_1_1)

        mask_1_2 = torch.logical_not(mask_1_1)
        ids_cur_1_2, deg_i_1_2 = ids_cur_1[mask_1_2], deg_i_1[mask_1_2]
        rand_i_1_2 = (torch.rand(deg_i_1_2.sum()) *
                      torch.repeat_interleave(dataset.deg[ids_cur_1_2], deg_i_1_2)).to(torch.long)
        idx_i_1_2 = rand_i_1_2 + torch.repeat_interleave(dataset.rowptr[ids_cur_1_2], deg_i_1_2)
        idx_i_1 = torch.cat([idx_i_1_1, idx_i_1_2])

        idx_i_2 = dataset.neigh_index(ids_cur_2)
        rowids_1, colids_1 = dataset.data.edge_index[:, idx_i_1]
        rowids_2, colids_2 = dataset.data.edge_index[:, idx_i_2]

        edges_row.append(torch.cat([rowids_1, rowids_2]))
        edges_col.append(torch.cat([colids_1, colids_2]))
        edges_weight.append(dataset.row_normalize(
            torch.ones_like(edges_row[-1]).view(-1, 1),
            edges_row[-1], num_nodes))


        if dataset.args.dataset == 'ogbn-papers100M':
            ids_cur = torch.LongTensor(pd.unique(edges_col[-1].numpy()))
            ids_record = torch.cat([ids_record, ids_cur])
        else:
            ids_record_i = torch.zeros(num_nodes, dtype=torch.bool)
            ids_record_i[edges_col[-1]] = True
            ids_cur = torch.where(ids_record_i)[0]
            ids_record[edges_col[-1]] = True

    ### reindex ids in top-down manner
    ########################################
    if dataset.args.dataset == 'ogbn-papers100M':
        ids_all = torch.LongTensor(pd.unique(ids_record.numpy()))
    else:
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