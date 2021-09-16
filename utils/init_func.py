import math
import scipy.sparse as sp
import numpy as np
import torch


from numba import njit
@njit
def neigh_index(rowptr_i, rowptr, deg):
    index = np.arange(deg.sum())
    for i in range(deg.shape[0]):
        index[rowptr_i[i]:rowptr_i[i+1]] = np.arange(deg[i]) + rowptr[i]
    return index

@njit
def unqiue_func(ids):
    result = []
    for i, val in enumerate(ids):
        if val:
            result.append(i)
    return result

@njit
def repeat_func(src, rowptr):
    result = np.arange(rowptr[-1])
    for i in range(rowptr.shape[0]):
        result[rowptr[i]:rowptr[i+1]] = src[i]
    return result

@njit
def choice_multi_range(c_range, s_num):
    index = np.arange(s_num.sum())
    shift = 0
    for i in range(c_range.shape[0]):
        # perm_i = np.random.permutation(c_range[i])
        perm_i = np.random.choice(c_range[i], s_num[i], replace=False)
        index[shift: shift + s_num[i]] = perm_i[:s_num[i]]
        shift += s_num[i]
    return index

@njit
def choice_multi_range_v2(rowptr, num_nb):
    index_nb = np.arange(num_nb.sum())
    shift = 0
    for i in range(num_nb.shape[0]):
        perm_i = np.random.choice(rowptr[i+1] - rowptr[i], num_nb[i], replace=False)
        index_nb[shift: shift + num_nb[i]] = rowptr[i] + perm_i
        shift += num_nb[i]
    return index_nb

@njit
def choice_multi_range_v3(num_nb, num_b, deg, rowptr):
    index_nb = np.arange(num_nb.sum())
    index_b = np.arange(num_b.sum())
    shift_nb, shift_b = 0, 0
    for i in range(num_nb.shape[0]):
        perm_i = np.random.permutation(deg[i])
        index_nb[shift_nb: shift_nb + num_nb[i]] = rowptr[i] + perm_i[:num_nb[i]]
        shift_nb += num_nb[i]
        index_b[shift_b: shift_b + num_b[i]] = rowptr[i] + perm_i[num_nb[i]:]
        shift_b += num_b[i]
    return index_nb, index_b

@njit
def choice_multi_range_multi_sample(c_range, s_num_1, s_num_2, rowptr):
    index_1 = np.arange(s_num_1 * c_range.shape[0])
    index_2 = np.arange(s_num_2 * c_range.shape[0])
    shift_1 = 0
    shift_2 = 0
    for i in range(c_range.shape[0]):
        # perm_i = np.random.permutation(c_range[i])
        perm_i = np.random.choice(c_range[i], s_num_1+s_num_2, replace=False)
        index_1[shift_1: shift_1 + s_num_1] = perm_i[:s_num_1] + rowptr[i]
        index_2[shift_2: shift_2 + s_num_2] = perm_i[s_num_1:s_num_1+s_num_2] + rowptr[i]
        shift_1 += s_num_1
        shift_2 += s_num_2
    return index_1, index_2


from torch_scatter import scatter_max, scatter_add
from torch_geometric.utils.num_nodes import maybe_num_nodes
def lap_normalize(src, index, num_nodes=None):
    num_nodes = maybe_num_nodes(index[0], num_nodes)
    # out = src / scatter_add(src, index[0], dim=0, dim_size=num_nodes)[index[0]]
    out = src / torch.sqrt(
        scatter_add(src, index[0], dim=0, dim_size=num_nodes)[index[0]] + 1e-16)
    out = out / torch.sqrt(
        scatter_add(src, index[1], dim=0, dim_size=num_nodes)[index[1]] + 1e-16)
    return out

def row_normalize(src, index, num_nodes=None):
    num_nodes = maybe_num_nodes(index[0], num_nodes)
    out = src / (scatter_add(src, index[0], dim=0, dim_size=num_nodes)[index[0]] + 1e-16)
    return out

def col_normalize(src, index, num_nodes=None):
    num_nodes = maybe_num_nodes(index[0], num_nodes)
    out = src / (scatter_add(src, index[0], dim=0, dim_size=num_nodes)[index[1]] + 1e-16)
    return out

def softmax(src, index, num_nodes=None):
    num_nodes = maybe_num_nodes(index, num_nodes)
    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)
    return out