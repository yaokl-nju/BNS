import torch
from utils.init_func import choice_multi_range_multi_sample
import pandas as pd
from torch_sparse import SparseTensor

def relabel_cpu(reindex, edges_col, oid_all):
    bool_id = torch.zeros(reindex.size(0), dtype=torch.bool)
    bool_id[edges_col] = True
    bool_id[oid_all] = True
    oid = torch.where(bool_id)[0]
    nid = reindex[oid]
    return oid[nid < 0]

def le(ids, deg, threshold):
    mask = deg < threshold
    return ids[mask], ids[torch.logical_not(mask)]

def ge(ids, deg, threshold):
    mask = deg > threshold
    return ids[mask], ids[torch.logical_not(mask)]

def leq(ids, deg, threshold):
    mask = deg <= threshold
    return ids[mask], ids[torch.logical_not(mask)]

def geq(ids, deg, threshold):
    mask = deg >= threshold
    return ids[mask], ids[torch.logical_not(mask)]

def BNS(ids_0, dataset):
    num_nodes = dataset.data.x.size(0)
    ids_cur = ids_0

    if dataset.args.dataset == 'ogbn-papers100M':
        ids_record_2 = torch.LongTensor([])
        ids_record_1 = ids_0
    else:
        ids_record_1 = torch.zeros(num_nodes, dtype=torch.bool)
        ids_record_1[ids_0] = True
        ids_record_2 = torch.zeros(num_nodes, dtype=torch.bool)
    edges_row, edges_col, edges_weight = [], [], []
    for i in range(dataset.args.layer_num):
        rho = dataset.args.rho
        s_n_nb = dataset.args.s_n_1 if i == 0 else dataset.args.s_n
        s_n_b = s_n_nb * rho

        ids_cur_1, ids_cur_2 = leq(ids_cur, dataset.deg[ids_cur], s_n_nb * (rho + 1))
        ids_cur_1_1, ids_cur_1_2 = ge(ids_cur_1, dataset.deg[ids_cur_1], s_n_nb)
        ids_cur_2_1, ids_cur_2_2 = le(ids_cur_2, dataset.deg[ids_cur_2], s_n_nb * (rho + 1) * 10)

        deg_i_1_1 = torch.zeros(ids_cur_1_1.size(0), dtype=torch.long) + s_n_nb
        idx_i_1_1_1, idx_i_1_1_2, weight_1_1_1, weight_1_1_2 = \
            dataset.neigh_index(ids_cur_1_1, deg_i_1_1)
        rowids_1_1_1, colids_1_1_1 = dataset.data.edge_index[:, idx_i_1_1_1]    # non-block
        rowids_1_1_2, colids_1_1_2 = dataset.data.edge_index[:, idx_i_1_1_2]    # block
        idx_i_1_2 = dataset.neigh_index(ids_cur_1_2)
        rowids_1_2, colids_1_2 = dataset.data.edge_index[:, idx_i_1_2]          # non-block

        idx_i_2_1_1, idx_i_2_1_2 = choice_multi_range_multi_sample(
            dataset.deg[ids_cur_2_1].numpy(), s_n_nb, s_n_b, dataset.rowptr[ids_cur_2_1].numpy())
        rand_i_2_2_1 = (torch.rand(s_n_nb * ids_cur_2_2.size(0)) *
                        torch.repeat_interleave(dataset.deg[ids_cur_2_2], s_n_nb)).to(torch.long)
        idx_i_2_2_1 = rand_i_2_2_1 + torch.repeat_interleave(dataset.rowptr[ids_cur_2_2], s_n_nb)
        rand_i_2_2_2 = (torch.rand(s_n_b * ids_cur_2_2.size(0)) *
                        torch.repeat_interleave(dataset.deg[ids_cur_2_2], s_n_b)).to(torch.long)
        idx_i_2_2_2 = rand_i_2_2_2 + torch.repeat_interleave(dataset.rowptr[ids_cur_2_2], s_n_b)
        idx_i_2_1 = torch.cat([torch.LongTensor(idx_i_2_1_1), idx_i_2_2_1])
        idx_i_2_2 = torch.cat([torch.LongTensor(idx_i_2_1_2), idx_i_2_2_2])
        rowids_2_1, colids_2_1 = dataset.data.edge_index[:, idx_i_2_1]           # non-block
        rowids_2_2, colids_2_2 = dataset.data.edge_index[:, idx_i_2_2]           # block


        weight_1_2 = torch.ones(rowids_1_2.size(0))
        weight_2_1 = torch.ones(rowids_2_1.size(0)) * rho
        weight_2_2 = torch.ones(rowids_2_2.size(0))

        rowid_i = torch.cat([rowids_1_1_1, rowids_1_2, rowids_2_1, rowids_1_1_2, rowids_2_2])
        colid_i = torch.cat([colids_1_1_1, colids_1_2, colids_2_1, colids_1_1_2, colids_2_2])
        weight_i = torch.cat([weight_1_1_1, weight_1_2, weight_2_1, weight_1_1_2, weight_2_2])
        size_1st = rowids_1_1_1.size(0) + rowids_1_2.size(0) + rowids_2_1.size(0)

        if i > 0:
            if dataset.args.dataset == 'ogbn-papers100M':
                selfloop_id = torch.LongTensor(pd.unique(ids_record_2.numpy()))
                ids_record_2 = selfloop_id
            else:
                selfloop_id = torch.where(ids_record_2)[0]
            rowid_i = torch.cat([rowid_i, ids_cur_2, selfloop_id])
            colid_i = torch.cat([colid_i, ids_cur_2, selfloop_id])
            if i == dataset.args.layer_num - 1:
                weight_i = torch.ones(rowid_i.size(0))
            else:
                weight_i = torch.cat([weight_i, torch.ones(ids_cur_2.size(0) + selfloop_id.size(0))])
        else:
            rowid_i = torch.cat([rowid_i, ids_cur_2])
            colid_i = torch.cat([colid_i, ids_cur_2])
            weight_i = torch.cat([weight_i, torch.ones(ids_cur_2.size(0))])

        edges_row.append(rowid_i)
        edges_col.append(colid_i)
        edges_weight.append(dataset.row_normalize(
            weight_i, edges_row[-1], num_nodes))

        if dataset.args.dataset == 'ogbn-papers100M':
            ids_record_2 = torch.cat([ids_record_2, edges_col[-1][size_1st:]])
            ids_record_1 = torch.cat([ids_record_1, ids_record_2])
            ids_cur = torch.LongTensor(pd.unique(edges_col[-1][:size_1st].numpy()))
            ids_record_1 = torch.cat([ids_record_1, ids_cur])
        else:
            ids_record_1[edges_col[-1]] = True
            ids_record_2[edges_col[-1][size_1st:]] = True
            ids_record_i_1 = torch.zeros(num_nodes, dtype=torch.bool)
            ids_record_i_1[edges_col[-1][:size_1st]] = True
            ids_cur = torch.where(ids_record_i_1)[0]

    ### reindex ids in top-down manner
    if dataset.args.dataset == 'ogbn-papers100M':
        ids_all = torch.LongTensor(pd.unique(ids_record_1.numpy()))
    else:
        ids_all = torch.where(ids_record_1)[0]
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
            if dataset.args.model != 'PNA':
                graph.append(torch.sparse.FloatTensor(edge_index_i, edges_weight[i].view(-1),
                                                      (oid_list[-1].size(0), offset)))
            else:
                graph.append(SparseTensor(row=edge_index_i[0], col=edge_index_i[1], value=edges_weight[i].view(-1),
                                          sparse_sizes=(oid_list[-1].size(0), offset)))
        else:
            graph.append(edge_index_i)
        if i < dataset.args.layer_num - 1:
            edge_row_i = reindex[edges_row_new[i + 1]]
        oid_list.append(ids_all[oid_all])
    idx_re = reindex[remap[ids_0]]
    graph.reverse()
    oid_list.reverse()
    return [graph, oid_list[0], ids_0, idx_re]