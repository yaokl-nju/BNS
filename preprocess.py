from ogb.nodeproppred import PygNodePropPredDataset
import os.path as osp
import numpy as np

import gc, time, joblib, torch
from torch_scatter import scatter_add
from utils.init_func import neigh_index
from utils.init_func import choice_multi_range, row_normalize
from torch_geometric.utils import add_self_loops
import pandas as pd

class Preprocess(PygNodePropPredDataset):
    def __init__(self, args, name, root='dataset', transform=None, pre_transform=None, meta_dict=None):
        self.name = name  ## original name, e.g., ogbl-ppa
        start_time = time.time()
        super(Preprocess, self).__init__(self.name, root, transform, pre_transform, meta_dict)
        self.num_nodes = self.data.y.size(0)
        ld_time = time.time() - start_time
        print('load data time: {:f}s'.format(ld_time))
        #######################################################
        #######################################################
        #######################################################
        self.args = args
        print("\tnum_nodes", self.data.y.size(0))
        self.load()

    def download(self):
        pass

    # ===============================================
    def load(self):
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
        print('preprocess step 3, time: {:f}s'.format(pd_time))
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

    def subgraph(self, ids):
        idx = self.neigh_index(ids)
        rows, cols = self.data.edge_index[:, idx]
        weights = self.data.edge_weight[idx]

        reindex = torch.zeros(self.num_nodes, dtype=torch.long) - 1
        reindex[ids] = torch.arange(ids.size(0))
        rows = reindex[rows]

        ids_map = torch.LongTensor(pd.unique(torch.cat([ids, cols]).numpy()))
        ids_col = ids_map[reindex[ids_map] < 0]
        ids_map = torch.cat([ids, ids_col])
        reindex[ids_col] = torch.arange(ids_col.size(0)) + ids.size(0)
        edge_index = torch.stack([rows, reindex[cols]], dim=0)
        graph = torch.sparse.FloatTensor(edge_index, weights.view(-1), (ids.size(0), ids_map.size(0)))
        return graph.to(self.args.device), ids_map

    def neigh_index(self, ids):
        deg = self.deg[ids]
        rowptr_i = deg.new_zeros(ids.shape[0] + 1)
        torch.cumsum(deg, 0, out=rowptr_i[1:])

        deg_sum = deg.sum()
        index = torch.arange(deg_sum)
        index -= torch.repeat_interleave(rowptr_i[:-1], deg)
        index += torch.repeat_interleave(self.rowptr[ids], deg)
        return index

    def row_normalize(self, src, row_index, num_nodes):
        out = src / (scatter_add(src, row_index, dim=0, dim_size=num_nodes)[row_index] + 1e-16)
        return out

    def save(self, x, layer=1):
        start_time = time.time()
        path = osp.join(self.root, 'processed/geometric_data_processed_{}.pt'.format(str(layer)))
        joblib.dump(x.numpy(), path, compress=3)
        print("save done, time={:f}s\n\n".format(time.time() - start_time))



from parse_conf import *
args.dataset = 'ogbn-products'
args.root = '/data/ssd/yaokl/ogbn'
dataset = Preprocess(args, name=args.dataset, root=args.root)
args.num_nodes = dataset.data.x.size(0)
args.bsize = 128 * 1024


start_time = time.time()
perm = torch.arange(args.num_nodes).split(args.bsize)
from tqdm import tqdm
x_gconv = []
for ids in tqdm(perm, ncols=70, leave=True):
    graph, ids_map = dataset.subgraph(ids)
    x_gconv.append((graph @ dataset.data.x[ids_map].to(args.device)).cpu())
    torch.cuda.empty_cache()
    del ids_map, graph
    gc.collect()
x = torch.cat(x_gconv, dim=0)
print("convx.size", x.size())
del x_gconv
gc.collect()
torch.cuda.empty_cache()
print("preprocess done, time={:f}s".format(time.time() - start_time))
dataset.save(x, 1)







