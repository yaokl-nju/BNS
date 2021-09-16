from ogb.nodeproppred import PygNodePropPredDataset
import os.path as osp
import numpy as np

import gc, time, pandas, joblib, torch
from torch_scatter import scatter_mean
from torch_scatter import scatter_add
from utils.init_func import neigh_index
from utils.init_func import choice_multi_range, row_normalize
from torch_geometric.utils import add_self_loops
from torch_sparse import SparseTensor


from global_val import *
import samplers

class ogbn_dataset(PygNodePropPredDataset):
    def __init__(self, args, name, root='dataset', transform=None, pre_transform=None, meta_dict=None):
        self.name = name  ## original name, e.g., ogbl-ppa
        start_time = time.time()
        super(ogbn_dataset, self).__init__(self.name, root, transform, pre_transform, meta_dict)
        self.num_nodes = self.data.y.size(0)
        ld_time = time.time() - start_time
        print('load data time: {:f}s'.format(ld_time))
        #######################################################
        #######################################################
        #######################################################
        self._get_idx_split()
        self.args = args
        self.batch = {}
        for phase in self.idx_dict.keys():
            bsize = self.idx_dict[phase].size(0) if self.args.bsize < 0 else self.args.bsize
            if phase == 'train':
                self.batch[phase] = self.idx_dict[phase].size(0) // bsize
            else:
                self.batch[phase] = int(np.ceil(self.idx_dict[phase].size(0) * 1.0 / bsize))

        print("\tnum_nodes", self.data.y.size(0))
        if args.dataset != 'ogbn-proteins':
            print("\tnum_features", self.data.x.size(1))
        print("\tnum_classes", self.num_classes)
        print("\tnum_train", self.idx_dict['train'].size(0))
        print("\tnum_valid", self.idx_dict['valid'].size(0))
        print("\tnum_test", self.idx_dict['test'].size(0))
        print("\ty.shape", self.data.y.shape)
        self.load()
        # self.process_for_LongTail()
        print("\tdeg: avg, median", self.deg.to(torch.float).mean(), self.deg.median())
        self.sampler = getattr(samplers, args.method)

    @property
    def processed_convfeat(self):
        path = osp.join(self.root, 'processed/geometric_data_processed_1.pt')
        return path

    def _get_idx_split(self):
        self.idx_dict = self.get_idx_split()

    '''
    must overload the funcs 'download' and 'process'
    '''
    def download(self):
        pass

    @property
    def num_classes(self):
        self.__num_classes__ = int(self.meta_info["num classes"])
        if self.args.dataset == 'ogbn-proteins':
            self.__num_classes__ = 112
        return self.__num_classes__

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

    # ===============================================
    def preprocess_for_dataset(self):
        start_time = time.time()
        if self.args.dataset == 'ogbn-papers100M':
            self.data.x = None
            gc.collect()
            self.data.x = torch.FloatTensor(joblib.load(self.processed_convfeat))
        if self.args.dataset == 'ogbn-proteins':
            self.data.x = scatter_mean(self.data.edge_attr, self.data.edge_index[0], dim=0, dim_size=self.num_nodes)
        print('preprocess step 1, time: {:f}s'.format(time.time() - start_time))
        gc.collect()

        start_time = time.time()
        if self.args.dataset == 'ogbn-papers100M':
            edge_index_inv = self.data.edge_index.repeat(1, 2)
            edge_index_inv[0, edge_index_inv.size(1) // 2:] = edge_index_inv[1, :edge_index_inv.size(1) // 2]
            edge_index_inv[1, edge_index_inv.size(1) // 2:] = edge_index_inv[0, :edge_index_inv.size(1) // 2]
            self.data.edge_index = edge_index_inv
            del edge_index_inv
            gc.collect()
        if self.args.dataset != 'ogbn-proteins':
            self.data.y = self.data.y.view(-1).to(torch.long)
        else:
            y_weight = self.data.y[self.idx_dict['train']].to(torch.float)
            pos_num = y_weight.sum(0).to(torch.float)
            neg_num = y_weight.size(0) - pos_num
            weight = neg_num / pos_num
            self.data.y_weight = weight
        print('preprocess step 2, time: {:f}s'.format(time.time() - start_time))

    def preprocess_for_samplers(self):
        if (self.args.method == 'LADIES' and self.args.dataset == 'ogbn-papers100M') or self.args.dataset == 'ogbn-proteins':
            self.graph_sp = SparseTensor(col=self.data.edge_index[1], rowptr=self.rowptr,
                                         value=self.data.edge_weight.view(-1),
                                         sparse_sizes=(self.num_nodes, self.num_nodes),
                                         is_sorted=True
                                         )
        start_time = time.time()
        # for LADIES
        selfmask = self.data.edge_index[0] == self.data.edge_index[1]
        selfweight = self.data.edge_weight[selfmask]
        selfedges = self.data.edge_index[0, selfmask]
        self.selfweight = selfweight[torch.argsort(selfedges)].view(-1)
        del selfmask, selfweight, selfedges
        gc.collect()

        if self.args.method == 'LADIES':
            node_probs = torch.zeros(self.num_nodes).type(torch.FloatTensor)
            node_probs.scatter_add_(0, self.data.edge_index[1], self.data.edge_weight.view(-1) ** 2)
            node_probs = node_probs / node_probs.sum()
            self.node_probs = node_probs
            del node_probs

        print('preprocess step 4, time: {:f}s'.format(time.time() - start_time))
        gc.collect()

    def load(self):
        self.preprocess_for_dataset()
        ### edge_index, edge_weight, deg, rowptr
        start_time = time.time()
        self.data.edge_index, _ = add_self_loops(self.data.edge_index, num_nodes=self.data.y.size(0))
        perm = torch.argsort(self.data.edge_index[0] * self.num_nodes + self.data.edge_index[1])
        self.data.edge_index = self.data.edge_index[:, perm]
        del perm
        gc.collect()
        self.data.edge_weight = row_normalize(
            torch.ones(self.data.edge_index.size(1), 1), self.data.edge_index, num_nodes=self.num_nodes)
        self.deg = torch.zeros(self.num_nodes, dtype=torch.long)
        self.deg.scatter_add_(0, self.data.edge_index[0], torch.ones(self.data.edge_index.size(1), dtype=torch.long))
        self.rowptr = torch.zeros(self.num_nodes + 1, dtype=torch.long)
        torch.cumsum(self.deg, 0, out=self.rowptr[1:])
        gc.collect()
        pd_time = time.time() - start_time
        print("\tgraph.nnz, average", self.data.edge_index.size(1), self.data.edge_index.size(1) // self.num_nodes)
        # print("\tedge_num", self.data.edge_index.size(1) // 2)
        print('preprocess step 3, time: {:f}s'.format(pd_time))
        self.preprocess_for_samplers()
    # ===============================================

    # ===============================================
    # funcs of sampling data
    def reset_iter(self):
        self.perm_idx = {}
        for phase in self.idx_dict.keys():
            self.perm_idx[phase] = self.iter_idx(phase)

    def iter_idx(self, phase):
        bsize = self.idx_dict[phase].size(0) if self.args.bsize < 0 else self.args.bsize
        if phase == 'train':
            perm = torch.randperm(self.idx_dict[phase].size(0)).split(bsize)
            perm = perm if self.idx_dict[phase].size(0) % bsize == 0 else perm[:-1]
        else:
            perm = torch.arange(self.idx_dict[phase].size(0)).split(bsize)
        for ids in perm:
            yield ids
        yield None

    def get_idx(self, phase):
        index = next(self.perm_idx[phase])
        if index is None:
            self.perm_idx[phase] = self.iter_idx(phase)
            index = next(self.perm_idx[phase])
        ids = self.idx_dict[phase][index]
        return ids

    def refresh_T1(self, T1):
        for i in range(len(T1)):
            self.T1[i] += T1[i]

    def sample(self, phase):
        if self.args.fast_version:
            while True:
                global batch
                if len(batch[phase]) > 0:
                    params = batch[phase].pop(0)
                    break
                else:
                    time.sleep(0.1)
            graph, ids_map, ids, idx = params
        else:
            ids = self.get_idx(phase)
            graph, ids_map, _, idx = self.sampler(ids, self)
        if isinstance(graph, list):
            graph = [gi.to(self.args.device) for gi in graph]
        else:
            graph = graph.to(self.args.device)
        lb = self.data.y[ids].to(self.args.device)
        sub_X = self.data.x[ids_map].to(self.args.device)
        return idx, lb, sub_X, graph

    def coalesce(self, index, value, m, n, op='add', fill_value=0):
        row, col = index
        val = torch.LongTensor(pandas.unique((row * n + col).numpy()))
        row = val // n
        col = val - row * n
        index = torch.stack([row, col], dim=0)
        return index, None

    def neigh_index(self, ids, sample_neigh_num_nb=None):
        deg = self.deg[ids]
        rowptr_i = deg.new_zeros(ids.shape[0] + 1)
        torch.cumsum(deg, 0, out=rowptr_i[1:])

        deg_sum = deg.sum()
        if self.args.dataset != 'ogbn-proteins':
            index = torch.arange(deg_sum)
            index -= torch.repeat_interleave(rowptr_i[:-1], deg)
            index += torch.repeat_interleave(self.rowptr[ids], deg)
        else:
            index = neigh_index(rowptr_i.numpy(), self.rowptr[ids].numpy(), self.deg[ids].numpy())
            index = torch.LongTensor(index)

        if sample_neigh_num_nb is None:
            return index
        else:
            idx_i_split = torch.LongTensor(choice_multi_range(deg.numpy(), sample_neigh_num_nb.numpy())) \
                          + torch.repeat_interleave(rowptr_i[:-1], sample_neigh_num_nb)

            mask = torch.zeros(index.size(0), dtype=torch.bool)
            mask[idx_i_split] = True
            sample_neigh_num_b = deg - sample_neigh_num_nb
            ratio = sample_neigh_num_b * 1.0 / sample_neigh_num_nb
            ratio[ratio < 1] = 1.0
            weight_1 = torch.repeat_interleave(torch.ones(ids.size(0)) * ratio, sample_neigh_num_nb)
            weight_2 = torch.repeat_interleave(torch.ones(ids.size(0)), sample_neigh_num_b)
            return index[mask], index[torch.logical_not(mask)], weight_1, weight_2

    def row_normalize(self, src, row_index, num_nodes):
        out = src / (scatter_add(src, row_index, dim=0, dim_size=num_nodes)[row_index] + 1e-16)
        return out
